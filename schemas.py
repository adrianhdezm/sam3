"""Request/response contract for POST /segment, shared by the local FastAPI app and the Runpod worker."""

from pydantic import BaseModel, Field


class SegmentRequest(BaseModel):
    image_base64: str = Field(
        ..., description="Base64-encoded image (PNG/JPEG/WebP...). A data-URL prefix is accepted."
    )
    text: str = Field(..., min_length=1, description="Concept to segment, e.g. 'building facade'.")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum detection score to keep an instance.")
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Probability cutoff used to binarize masks.")


class SegmentItem(BaseModel):
    score: float
    box: list[float] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] in pixels.")
    polygon: list[list[float]] = Field(..., description="Outer contour [[x, y], ...] in pixels; [] if degenerate.")


class SegmentResponse(BaseModel):
    items: list[SegmentItem]
