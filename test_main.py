import io
import glob

import pytest
from fastapi.testclient import TestClient

from main import app
from config import config


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _sample_eval_tile():
    candidates = sorted(
        glob.glob(str(config.candidate_tiles_dir.parent / "eval_set" / "*.png"))
    )
    if not candidates:
        pytest.skip("eval_set not found")
    return candidates[0]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_classify_real_tile_returns_valid_prediction(client):
    path = _sample_eval_tile()
    with open(path, "rb") as f:
        r = client.post("/tiles/classify", files={"file": (path, f, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["predicted_label"], str) and body["predicted_label"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["class_scores"], dict)
    assert abs(sum(body["class_scores"].values()) - 1.0) < 0.01
    assert "needs_review" in body


def test_classify_rejects_unreadable_file(client):
    bad_file = io.BytesIO(b"this is not an image")
    r = client.post(
        "/tiles/classify", files={"file": ("bad.png", bad_file, "image/png")}
    )
    assert r.status_code == 400


def test_classify_stores_and_is_queryable(client):
    path = _sample_eval_tile()
    with open(path, "rb") as f:
        r = client.post("/tiles/classify", files={"file": (path, f, "image/png")})
    result_id = r.json()["id"]

    r2 = client.get("/tiles", params={"limit": 500})
    assert r2.status_code == 200
    ids = [row["id"] for row in r2.json()]
    assert result_id in ids


def test_query_filter_by_label(client):
    r = client.get("/tiles", params={"label": "Forest", "limit": 500})
    assert r.status_code == 200
    for row in r.json():
        assert row["predicted_label"] == "Forest"                               