"""Local HTTP server: POST /segment. Runs on CUDA, Apple Silicon (MPS) or CPU.

    uv run uvicorn app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException

import segmenter
from schemas import SegmentItem, SegmentRequest, SegmentResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the model at startup so the first request is fast and misconfiguration fails early.
    if os.environ.get("SAM3_LAZY_LOAD") != "1":
        segmenter.get_segmenter()
    yield


app = FastAPI(title="SAM3 segmentation", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(segmenter.pick_device())}


# Sync handler on purpose: FastAPI runs it in a worker thread, so a multi-second
# inference does not block the event loop (and /health keeps answering).
@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest) -> SegmentResponse:
    try:
        image = segmenter.decode_image(req.image_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    detections = segmenter.get_segmenter().segment(image, req.text, req.threshold, req.mask_threshold)
    return SegmentResponse(items=[SegmentItem(**asdict(d)) for d in detections])
