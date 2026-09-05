"""Runpod Flash load-balanced GPU endpoint exposing POST /segment.

    uv run flash dev      # dev endpoints on Runpod (prefixed "live-"), router at http://localhost:8888
    uv run flash deploy   # production; prints the https://<id>.api.runpod.ai base URL

Flash ships each handler's source to the worker on its own (at least in `flash dev`), so everything a
handler uses -- third-party packages *and* local modules -- must be imported inside its body.
"""

import os

from runpod_flash import DataCenter, Endpoint, GpuGroup, NetworkVolume, load_dotenv

load_dotenv()  # local .env -> HF_TOKEN etc. Values only reach the worker via `env=` below.

# ~3.5 GB of weights: persist the HF cache on a network volume so cold starts don't re-download.
# A volume pins the endpoint to its datacenter; override with SAM3_DATACENTER (see DataCenter enum).
model_cache = NetworkVolume(
    name="sam3-model-cache",
    size=20,
    datacenter=DataCenter[os.environ.get("SAM3_DATACENTER", "EU_RO_1")],
)

api = Endpoint(
    name="sam3-segment",
    gpu=[GpuGroup.ADA_24, GpuGroup.AMPERE_24],  # 24 GB is ample for SAM3 in bf16
    workers=(0, 2),
    idle_timeout=120,
    execution_timeout_ms=120_000,
    # The Flash GPU image ships torch 2.9.1+cu128 (Python 3.12) but no torchvision, which the SAM3 image
    # processor needs. Pin the torchvision that targets that exact torch so pip doesn't pull a new torch.
    dependencies=[
        "torchvision==0.24.1",
        "transformers>=5.16",
        "accelerate",
        "pillow",
        "numpy",
        "opencv-python-headless",
    ],
    volume=model_cache,
    env={
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),  # facebook/sam3 is gated
        "HF_HOME": "/runpod-volume/huggingface",
        "SAM3_DTYPE": os.environ.get("SAM3_DTYPE", "bfloat16"),
    },
)


@api.get("/health")
async def health() -> dict:
    """Liveness plus what actually runs on the worker: pinned deps depend on the image's torch."""
    import importlib.metadata as md
    import os
    import time

    import torch

    def ver(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    try:
        uptime = round(time.time() - os.stat("/proc/self").st_ctime)  # Linux worker; absent on macOS
    except OSError:
        uptime = None

    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {p: ver(p) for p in ("torch", "torchvision", "transformers")},
        "worker_uptime_s": uptime,
    }


# Flat parameters on purpose: both the `flash dev` router and the production LB handler build the
# JSON body model from the handler signature, so this yields exactly the documented flat request
# ({"image_base64": ..., "text": ..., ...}). Range validation is re-applied via SegmentRequest.
@api.post("/segment")
async def segment(image_base64: str, text: str, threshold: float = 0.5, mask_threshold: float = 0.5) -> dict:
    import asyncio
    from dataclasses import asdict

    from fastapi import HTTPException
    from pydantic import ValidationError

    import segmenter
    from schemas import SegmentItem, SegmentRequest, SegmentResponse

    try:
        req = SegmentRequest(image_base64=image_base64, text=text, threshold=threshold, mask_threshold=mask_threshold)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors(include_url=False)) from e
    try:
        image = segmenter.decode_image(req.image_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    model = segmenter.get_segmenter()
    detections = await asyncio.to_thread(model.segment, image, req.text, req.threshold, req.mask_threshold)
    return SegmentResponse(items=[SegmentItem(**asdict(d)) for d in detections]).model_dump()
