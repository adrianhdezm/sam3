"""Runpod Flash load-balanced GPU endpoint exposing POST /segment.

    uv run flash dev      # dev endpoints on Runpod (prefixed "live-"), router at http://localhost:8888
    uv run flash deploy   # production; prints the https://<id>.api.runpod.ai base URL

Per Flash rules, third-party packages are imported inside the handler, not at module top.
"""

import asyncio
import os

from runpod_flash import DataCenter, Endpoint, GpuGroup, NetworkVolume, load_dotenv

from schemas import SegmentItem, SegmentRequest, SegmentResponse

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
    # torch is preinstalled in the Flash GPU image (Python 3.12); everything else we need:
    dependencies=["transformers>=5.16", "accelerate", "pillow", "numpy", "opencv-python-headless"],
    volume=model_cache,
    env={
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),  # facebook/sam3 is gated
        "HF_HOME": "/runpod-volume/huggingface",
        "SAM3_DTYPE": os.environ.get("SAM3_DTYPE", "bfloat16"),
    },
)


@api.get("/health")
async def health() -> dict:
    import torch

    return {"status": "ok", "cuda": torch.cuda.is_available()}


# A single Pydantic parameter means the JSON body is the request itself (no wrapper).
@api.post("/segment")
async def segment(req: SegmentRequest) -> dict:
    from dataclasses import asdict

    from fastapi import HTTPException

    import segmenter

    try:
        image = segmenter.decode_image(req.image_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    model = segmenter.get_segmenter()
    detections = await asyncio.to_thread(model.segment, image, req.text, req.threshold, req.mask_threshold)
    return SegmentResponse(items=[SegmentItem(**asdict(d)) for d in detections]).model_dump()
