# sam3 — text-prompted image segmentation API

`POST /segment` takes a base64 image and a short noun phrase ("building facade") and returns every
matching instance as a score, an `xyxy` box and an outer-contour polygon, all in original-image pixels.
Backed by [facebook/sam3](https://huggingface.co/facebook/sam3) via `transformers`.

The same inference code runs in two places:

| | file | runs on | how |
|---|---|---|---|
| Local server | `app.py` (FastAPI) | CUDA, Apple Silicon (MPS) or CPU | `uv run uvicorn app:app` |
| Production | `lb_worker.py` (Runpod Flash load-balanced endpoint) | Runpod serverless GPUs | `uv run flash deploy` |

```
schemas.py      request/response models (shared contract)
segmenter.py    model loading, inference, mask -> polygon (no web framework)
app.py          local FastAPI server
lb_worker.py    Runpod Flash endpoint definition
scripts/smoke_client.py   end-to-end client, can draw an overlay
tests/          unit tests (no weights needed)
```

## API

`POST /segment`

```json
{
  "image_base64": "<base64 or data URL>",
  "text": "building facade",
  "threshold": 0.5,
  "mask_threshold": 0.5
}
```

`threshold` is the minimum detection score to keep an instance; `mask_threshold` binarizes the mask
probabilities. Both default to `0.5`.

```json
{
  "items": [
    { "score": 0.91, "box": [120.5, 80.0, 980.2, 1450.7], "polygon": [[120.5, 90.1], [140.2, 85.4], ...] }
  ]
}
```

- Items are sorted by score, descending.
- `polygon` is the **outer contour of the largest connected component** of the instance mask, simplified
  with Douglas-Peucker (1 px tolerance, `SAM3_POLYGON_TOLERANCE`). Holes and disconnected fragments are
  dropped; `[]` if the mask is degenerate.
- Errors: `400` bad/undecodable image, `422` schema violation.

`GET /health` returns `{"status": "ok", ...}`.

## Prerequisites

1. **Hugging Face access** — `facebook/sam3` is gated. Accept the terms on the model page, then either
   `uv run hf auth login` or put `HF_TOKEN=hf_...` in `.env` (see `.env.example`).
2. `uv` and Python 3.12 (`uv` installs it; the project pins 3.12 because `runpod-flash` needs `<3.14`).

```bash
uv sync
```

## Run locally (Apple M-series or CUDA)

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

The model (~3.5 GB) downloads on first start into the HF cache and loads before the server accepts
requests. Device is auto-picked: `cuda` → `mps` → `cpu` (override with `SAM3_DEVICE`); dtype is bf16 on CUDA
and fp32 elsewhere (`SAM3_DTYPE`).

```bash
# quick check with an overlay image
uv run python scripts/smoke_client.py photo.jpg "building facade" --out overlay.png

# or raw curl
python3 -c 'import base64,json;print(json.dumps({"image_base64":base64.b64encode(open("photo.jpg","rb").read()).decode(),"text":"building facade"}))' \
  | curl -s -X POST localhost:8000/segment -H 'Content-Type: application/json' -d @- | head -c 600
```

Interactive docs at http://localhost:8000/docs.

Tests (no model download required):

```bash
uv run pytest
```

## Deploy to Runpod (Flash)

Flash turns `lb_worker.py` into a load-balanced serverless endpoint. Note that **`flash dev` also runs
your functions on Runpod GPUs** (dev endpoints prefixed `live-`), with a local router on `:8888`; only the
FastAPI app above runs on your own machine.

```bash
cp .env.example .env        # fill HF_TOKEN (and RUNPOD_API_KEY, or use `flash login`)
uv run flash login

# 1. dev loop: real GPUs, hot reload, Swagger at http://localhost:8888/docs
uv run flash dev --auto-provision
uv run python scripts/smoke_client.py photo.jpg "window" --url http://localhost:8888/lb_worker

# 2. production
uv run flash deploy         # prints the endpoint URL, e.g. https://<id>.api.runpod.ai
uv run python scripts/smoke_client.py photo.jpg "window" --url https://<id>.api.runpod.ai --api-key $RUNPOD_API_KEY

# tear down
uv run flash undeploy sam3-segment
```

Deployed calls need `Authorization: Bearer $RUNPOD_API_KEY`; the body is the plain request JSON (no
`{"input": ...}` wrapper — that is only for queue-based endpoints). During `flash dev` the routes are
mounted under the module name: `http://localhost:8888/lb_worker/segment`. Two `flash dev`-only quirks: handler errors come back as
`500 Remote execution failed: 400: ...` (production returns the real `400`/`422`), and every name a
handler uses must be imported inside its body, because dev ships only the function source to the worker.
Also, `flash dev` hot-installs dependency changes into the *running* worker; transformers caches its
"is torchvision available" check per process, so after adding a dependency recycle the worker with
`uv run flash undeploy live-sam3-segment --force` and restart `flash dev` (the volume/cache survives).

What `lb_worker.py` configures (edit there):

- **GPU**: `GpuGroup.ADA_24` with `AMPERE_24` fallback; `workers=(0, 2)` scales to zero, `idle_timeout=120`.
  Measured on an RTX 4090: ~45 s cold start, ~25 s first request (model load from the volume), then ~1.5 s
  per request end to end.
- **Model cache**: a 20 GB `NetworkVolume` mounted at `/runpod-volume`, `HF_HOME` pointed at it, so only the
  very first cold start downloads the weights. A volume pins the endpoint to one datacenter
  (`SAM3_DATACENTER`, default `EU_RO_1`) — pick one close to you with GPU availability.
- **Secrets**: `.env` is local-only; `HF_TOKEN` reaches workers only because it is passed through `env=`.
- **Dependencies**: the Flash GPU image ships torch 2.9.1+cu128 but no torchvision, so `torchvision==0.24.1`
  (the build for that torch) is pinned alongside `transformers`, `opencv-python-headless`, etc. in
  `dependencies=[...]`. If Runpod bumps the image's torch, `/health` reports the versions — re-pin to match.

## Environment variables

| var | default | purpose |
|---|---|---|
| `HF_TOKEN` | — | access to the gated model |
| `SAM3_MODEL_ID` | `facebook/sam3` | alternative checkpoint |
| `SAM3_DEVICE` | auto | force `cuda` / `mps` / `cpu` |
| `SAM3_DTYPE` | bf16 on CUDA, fp32 otherwise | `bfloat16`, `float16`, `float32` |
| `SAM3_POLYGON_TOLERANCE` | `1.0` | polygon simplification in px, `0` = none |
| `SAM3_LAZY_LOAD` | unset | `1` = don't load the model at server startup |
| `SAM3_DATACENTER` | `EU_RO_1` | Runpod network-volume datacenter |
