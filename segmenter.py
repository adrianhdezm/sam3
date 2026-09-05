"""SAM3 text-prompted instance segmentation: model loading, inference, mask -> polygon.

Framework-free so the same code serves the local FastAPI app and the Runpod worker.
"""

from __future__ import annotations

import base64
import binascii
import importlib.util
import io
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass

# MPS lacks a few ops; let PyTorch fall back to CPU for those instead of raising.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
# Douglas-Peucker epsilon for polygon simplification, in pixels. 0 disables simplification.
POLYGON_TOLERANCE_PX = float(os.environ.get("SAM3_POLYGON_TOLERANCE", "1.0"))
# Must match the torch preinstalled in the Runpod Flash GPU image (2.9.1+cu128); see lb_worker.py.
TORCHVISION_PIN = os.environ.get("SAM3_TORCHVISION_PIN", "torchvision==0.24.1")


@dataclass
class Detection:
    score: float
    box: list[float]  # [x1, y1, x2, y2], original-image pixels
    polygon: list[list[float]]  # [[x, y], ...], original-image pixels


def pick_device() -> torch.device:
    forced = os.environ.get("SAM3_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    forced = os.environ.get("SAM3_DTYPE")
    if forced:
        return getattr(torch, forced)
    # bf16 halves memory/bandwidth on CUDA; stay in fp32 on MPS/CPU for numerical safety.
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def decode_image(image_base64: str) -> Image.Image:
    """Decode a base64 (or data-URL) string into an RGB PIL image. Raises ValueError on bad input."""
    s = image_base64.strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    s = "".join(s.split())  # tolerate line-wrapped base64
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image_base64 is not valid base64: {e}") from e
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:  # PIL raises a mix of UnidentifiedImageError/OSError/ValueError
        raise ValueError(f"image_base64 does not decode to a supported image: {e}") from e
    # Honour EXIF orientation so boxes/polygons line up with what the user sees.
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def mask_to_polygon(mask: np.ndarray, tolerance: float = POLYGON_TOLERANCE_PX) -> list[list[float]]:
    """Outer contour of the largest connected component of a binary mask, simplified.

    One polygon per instance: holes and smaller disconnected fragments are dropped.
    Returns [] for an empty or degenerate (<3 vertices) mask.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if tolerance > 0:
        contour = cv2.approxPolyDP(contour, tolerance, closed=True)
    pts = contour.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return []
    return pts.tolist()


def _pip_install(spec: str) -> None:
    """Install a wheel into the running interpreter, trying pip, then ensurepip+pip, then uv."""
    attempts = [
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", spec],
        [sys.executable, "-m", "ensurepip", "--upgrade"],  # bootstraps pip; next attempt retries it
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", spec],
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", spec],
    ]
    errors = []
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            errors.append(f"{' '.join(cmd[:3])}: {getattr(e, 'stderr', None) or e}".strip()[:400])
            continue
        importlib.invalidate_caches()
        if importlib.util.find_spec(spec.split("==")[0]) is not None:
            return
    raise RuntimeError(f"could not install {spec}: " + " | ".join(errors))


def ensure_torchvision() -> None:
    """Install torchvision if absent. Sam3ImageProcessor requires it, but Runpod Flash strips torch*
    packages from deploy bundles (assuming the base image provides them) while the LB GPU image ships
    torch without torchvision. Must run before transformers is imported: it caches the availability check.
    """
    if importlib.util.find_spec("torchvision") is not None:
        return
    logger.warning("torchvision not found; installing %s", TORCHVISION_PIN)
    _pip_install(TORCHVISION_PIN)


class Sam3Segmenter:
    def __init__(self, model_id: str = MODEL_ID, device: torch.device | None = None, dtype: torch.dtype | None = None):
        ensure_torchvision()
        from transformers import Sam3Model, Sam3Processor  # heavy import kept local

        self.device = device or pick_device()
        self.dtype = dtype or pick_dtype(self.device)
        logger.info("Loading %s on %s (%s)", model_id, self.device, self.dtype)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        # One inference at a time per process: the accelerator is the bottleneck anyway,
        # and it keeps concurrent HTTP requests from interleaving on MPS.
        self._infer_lock = threading.Lock()

    @torch.inference_mode()
    def segment(
        self, image: Image.Image, text: str, threshold: float = 0.5, mask_threshold: float = 0.5
    ) -> list[Detection]:
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        original_sizes = inputs.get("original_sizes").tolist()
        inputs = inputs.to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with self._infer_lock:
            outputs = self.model(**inputs)
            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=mask_threshold, target_sizes=original_sizes
            )[0]
            scores = results["scores"].float().cpu().numpy()
            boxes = results["boxes"].float().cpu().numpy()
            masks = results["masks"].cpu().numpy()

        detections = [
            Detection(
                score=round(float(score), 4),
                box=[round(float(v), 2) for v in box],
                polygon=mask_to_polygon(mask),
            )
            for score, box, mask in zip(scores, boxes, masks)
        ]
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


_instance: Sam3Segmenter | None = None
_instance_lock = threading.Lock()


def get_segmenter() -> Sam3Segmenter:
    """Process-wide singleton; the model is loaded on first use and reused across requests."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Sam3Segmenter()
    return _instance
