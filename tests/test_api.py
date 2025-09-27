import io
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app


def make_dummy_image_bytes(width: int = 32, height: int = 32) -> bytes:
    img = Image.new("RGB", (width, height), color=(123, 222, 64))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_predict_dummy(monkeypatch):
    # Monkeypatch model to avoid heavy load
    from backend.app import main as m

    class DummyBoxes:
        def __init__(self):
            import numpy as np
            self.xyxy = type("_", (), {"cpu": lambda self: type("_2", (), {"numpy": lambda self: np.array([[1, 2, 3, 4]])})()})()
            self.conf = type("_", (), {"cpu": lambda self: type("_2", (), {"numpy": lambda self: np.array([0.9])})()})()
            self.cls = type("_", (), {"cpu": lambda self: type("_2", (), {"numpy": lambda self: np.array([0])})()})()

    class DummyResult:
        def __init__(self):
            self.boxes = DummyBoxes()

    class DummyModel:
        names = {0: "rust"}

        def predict(self, source, verbose=False):
            return [DummyResult()]

    monkeypatch.setattr(m, "get_model", lambda: DummyModel())

    client = TestClient(app)
    img_bytes = make_dummy_image_bytes()
    files = {"image": ("x.jpg", img_bytes, "image/jpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    js = r.json()
    assert "predictions" in js
    assert len(js["predictions"]) == 1
    pred = js["predictions"][0]
    assert pred["class_name"] == "rust"
    assert 0 <= pred["confidence"] <= 1
    assert len(pred["bbox"]) == 4


