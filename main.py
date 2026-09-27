import io
from typing import Optional

import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image

from config import config
from features import extract_features
from storage import init_db, insert_result, insert_failure, list_results

app = FastAPI(title="Offline Tile Classifier — thin slice")

_model_artifact = None


@app.on_event("startup")
def startup():
    global _model_artifact
    init_db()
    if not config.model_artifact_path.exists():
        raise RuntimeError(
            f"No model artifact at {config.model_artifact_path}. Run train.py first."
        )
    _model_artifact = joblib.load(config.model_artifact_path)


@app.post("/tiles/classify")
async def classify_tile(file: UploadFile = File(...)):
    if _model_artifact is None:
        raise HTTPException(503, "Model not loaded")

    filename = file.filename or "unknown"
    model_version = str(config.model_artifact_path.name)
    feature_version = _model_artifact["feature_extractor_version"]

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as e:
        insert_failure(filename, model_version, feature_version,
                        error_message=f"unreadable image: {e}")
        raise HTTPException(400, "Could not read uploaded file as an image")

    try:
        feats = extract_features(image).reshape(1, -1)
        clf = _model_artifact["model"]
        encoder = _model_artifact["label_encoder"]

        proba = clf.predict_proba(feats)[0]
        best_idx = proba.argmax()
        label = encoder.inverse_transform([best_idx])[0]
        confidence = float(proba[best_idx])
        needs_review = confidence < config.low_confidence_threshold
        class_scores = {
            cls: round(float(p), 4)
            for cls, p in zip(encoder.classes_, proba)
        }
    except Exception as e:
        insert_failure(filename, model_version, feature_version,
                        error_message=f"inference error: {e}")
        raise HTTPException(500, "Classification failed")

    result_id = insert_result(
        filename=filename,
        predicted_label=label,
        confidence=confidence,
        class_scores=class_scores,
        needs_review=needs_review,
        model_version=model_version,
        feature_extractor_version=feature_version,
    )

    return {
        "id": result_id,
        "filename": filename,
        "predicted_label": label,
        "confidence": round(confidence, 4),
        "class_scores": class_scores,
        "needs_review": needs_review,
    }


@app.get("/tiles")
def query_tiles(
    label: Optional[str] = Query(None),
    needs_review: Optional[bool] = Query(None),
    limit: int = Query(50, le=500),
):
    return list_results(label=label, needs_review=needs_review, limit=limit)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_artifact is not None}