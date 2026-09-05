"""Unit tests that do not need the model weights: geometry, decoding, and the HTTP contract."""

import base64
import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ["SAM3_LAZY_LOAD"] = "1"  # don't load weights when importing the app

import app as app_module  # noqa: E402
import segmenter  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def png_b64(w=64, h=48, color=(200, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --- mask_to_polygon -----------------------------------------------------------------------

def test_rectangle_mask_gives_four_corners():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 10:50] = True  # rows 20..59, cols 10..49
    poly = segmenter.mask_to_polygon(mask)
    assert len(poly) == 4
    xs = {p[0] for p in poly}
    ys = {p[1] for p in poly}
    assert xs == {10.0, 49.0}
    assert ys == {20.0, 59.0}


def test_empty_mask_gives_empty_polygon():
    assert segmenter.mask_to_polygon(np.zeros((10, 10), dtype=bool)) == []


def test_degenerate_line_mask_gives_empty_polygon():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 2:8] = True
    assert segmenter.mask_to_polygon(mask) == []


def test_largest_component_wins():
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:5, 0:5] = True  # small speck
    mask[40:90, 40:90] = True  # main blob
    poly = segmenter.mask_to_polygon(mask)
    assert all(40 <= x <= 89 and 40 <= y <= 89 for x, y in poly)


def test_uint8_mask_from_model_is_accepted():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 1
    assert len(segmenter.mask_to_polygon(mask)) == 4


# --- ensure_torchvision ------------------------------------------------------------------

def test_ensure_torchvision_noop_when_present(monkeypatch):
    calls = []
    monkeypatch.setattr(segmenter.subprocess, "run", lambda *a, **k: calls.append(a))
    segmenter.ensure_torchvision()  # torchvision is installed in the dev env
    assert calls == []


def test_ensure_torchvision_installs_pin_without_deps(monkeypatch):
    calls = []
    # torchvision "appears" only after an install command has run
    monkeypatch.setattr(segmenter.importlib.util, "find_spec", lambda name: object() if calls else None)
    monkeypatch.setattr(segmenter.subprocess, "run", lambda cmd, **k: calls.append((cmd, k)))
    segmenter.ensure_torchvision()
    ((cmd, kwargs),) = calls
    assert cmd[:4] == [segmenter.sys.executable, "-m", "pip", "install"]
    assert "--no-deps" in cmd and cmd[-1] == segmenter.TORCHVISION_PIN
    assert kwargs.get("check") is True


def test_ensure_torchvision_falls_back_and_reports_all_failures(monkeypatch):
    monkeypatch.setattr(segmenter.importlib.util, "find_spec", lambda name: None)

    def fail(cmd, **k):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(segmenter.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="could not install torchvision") as ei:
        segmenter.ensure_torchvision()
    assert "uv pip install" in str(ei.value)  # the last fallback was tried


# --- decode_image --------------------------------------------------------------------------

def test_decode_plain_base64():
    img = segmenter.decode_image(png_b64(64, 48))
    assert img.mode == "RGB" and img.size == (64, 48)


def test_decode_data_url_and_wrapped_lines():
    b64 = png_b64()
    wrapped = "\n".join(b64[i : i + 40] for i in range(0, len(b64), 40))
    img = segmenter.decode_image("data:image/png;base64," + wrapped)
    assert img.size == (64, 48)


def test_decode_rejects_garbage():
    with pytest.raises(ValueError, match="not valid base64"):
        segmenter.decode_image("!!!not base64!!!")
    with pytest.raises(ValueError, match="supported image"):
        segmenter.decode_image(base64.b64encode(b"hello world").decode())


# --- HTTP contract (model stubbed) -------------------------------------------------------

class FakeSegmenter:
    def __init__(self):
        self.calls = []

    def segment(self, image, text, threshold, mask_threshold):
        self.calls.append((image.size, text, threshold, mask_threshold))
        return [
            segmenter.Detection(score=0.91, box=[1.0, 2.0, 30.0, 40.0], polygon=[[1.0, 2.0], [30.0, 2.0], [30.0, 40.0]]),
        ]


@pytest.fixture
def client(monkeypatch):
    fake = FakeSegmenter()
    monkeypatch.setattr(segmenter, "get_segmenter", lambda: fake)
    with TestClient(app_module.app) as c:
        c.fake = fake
        yield c


def test_segment_contract(client):
    r = client.post("/segment", json={"image_base64": png_b64(), "text": "building facade"})
    assert r.status_code == 200, r.text
    assert r.json() == {"items": [{"score": 0.91, "box": [1.0, 2.0, 30.0, 40.0], "polygon": [[1.0, 2.0], [30.0, 2.0], [30.0, 40.0]]}]}
    # defaults applied and forwarded
    assert client.fake.calls == [((64, 48), "building facade", 0.5, 0.5)]


def test_segment_forwards_thresholds(client):
    client.post("/segment", json={"image_base64": png_b64(), "text": "car", "threshold": 0.3, "mask_threshold": 0.7})
    assert client.fake.calls[-1][2:] == (0.3, 0.7)


def test_segment_bad_image_is_400(client):
    r = client.post("/segment", json={"image_base64": "@@@", "text": "car"})
    assert r.status_code == 400
    assert "base64" in r.json()["detail"]


def test_segment_validation_is_422(client):
    assert client.post("/segment", json={"image_base64": png_b64(), "text": "car", "threshold": 1.5}).status_code == 422
    assert client.post("/segment", json={"image_base64": png_b64(), "text": ""}).status_code == 422
    assert client.post("/segment", json={"text": "car"}).status_code == 422


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
