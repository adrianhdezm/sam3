"""End-to-end check against a running server (local FastAPI, flash dev router, or deployed Runpod URL).

    uv run python scripts/smoke_client.py photo.jpg "building facade" --out overlay.png
    uv run python scripts/smoke_client.py photo.jpg "window" --url https://<id>.api.runpod.ai --api-key $RUNPOD_API_KEY
"""

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("text")
    p.add_argument("--url", default="http://localhost:8000", help="server base URL")
    p.add_argument("--api-key", default=None, help="Runpod API key (adds Authorization: Bearer)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--out", default=None, help="write an overlay PNG with boxes and polygons")
    args = p.parse_args()

    with open(args.image, "rb") as f:
        payload = {
            "image_base64": base64.b64encode(f.read()).decode(),
            "text": args.text,
            "threshold": args.threshold,
            "mask_threshold": args.mask_threshold,
        }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    req = urllib.request.Request(
        args.url.rstrip("/") + "/segment", data=json.dumps(payload).encode(), headers=headers, method="POST"
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    dt = time.perf_counter() - t0

    items = body["items"]
    print(f"{len(items)} item(s) for {args.text!r} in {dt:.2f}s")
    for i, it in enumerate(items):
        print(f"  #{i} score={it['score']:.3f} box={[round(v) for v in it['box']]} polygon_pts={len(it['polygon'])}")

    if args.out:
        img = Image.open(args.image).convert("RGB")
        draw = ImageDraw.Draw(img)
        for it in items:
            draw.rectangle(it["box"], outline=(255, 0, 0), width=3)
            if len(it["polygon"]) >= 3:
                draw.polygon([tuple(pt) for pt in it["polygon"]], outline=(0, 255, 0), width=2)
            draw.text((it["box"][0] + 4, it["box"][1] + 4), f"{it['score']:.2f}", fill=(255, 255, 0))
        img.save(args.out)
        print(f"overlay written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
