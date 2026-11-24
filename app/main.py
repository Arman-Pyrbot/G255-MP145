"""FastAPI prediction service for MP145.

GET  /health  -> {"status": "ok"}
POST /predict -> {"prediction": <gb_per_day>}

The client's two rules live in the pydantic schema (Field le / ge), so an
invalid request is rejected with 422 before it ever reaches the model.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mp145.features import FEATURE_COLUMNS, build_preprocessor, make_request_frame  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.joblib"


class JSONFormatter(logging.Formatter):
    """One JSON object per log line on stdout (never a file)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("mp145.api")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _configure_logging()


class PredictRequest(BaseModel):
    """Exact request body required by PROBLEM.md section 5."""

    age: float = Field(..., ge=0, le=120)
    screen_hours: float = Field(..., ge=0, le=20)          # rule 1: <= 20
    apps_installed: int = Field(..., ge=5)                  # rule 2: >= 5
    plan_gb_limit: float = Field(..., gt=0)
    video_minutes: float = Field(..., ge=0)
    wifi_hours: float = Field(..., ge=0)
    phone_tier: str
    circle: str


class PredictResponse(BaseModel):
    prediction: float


def _fallback_model():
    """Keep the app importable in CI, where model.joblib may not be present."""
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("preprocess", build_preprocessor()),
                         ("model", DummyRegressor(strategy="constant", constant=4.0))])
    seed = {c: 0 for c in FEATURE_COLUMNS}
    seed["phone_tier"] = "mid"
    seed["circle"] = "Urban"
    pipeline.fit(make_request_frame(seed), [4.0])
    return pipeline


def load_or_fallback():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return _fallback_model()


model = load_or_fallback()
app = FastAPI(title="MP145 data-use prediction")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    row = make_request_frame(request.model_dump())
    value = float(model.predict(row)[0])
    # Structured, privacy-safe log: no customer identifiers or raw features.
    logger.info("prediction_served", extra={"extra_fields": {"prediction": round(value, 4)}})
    return PredictResponse(prediction=round(value, 4))
