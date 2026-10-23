"""HTTP model-serving API for artifacts created by train_model.py."""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from model_core import load_model_bundle

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model.joblib"))


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model_bundle = load_model_bundle(MODEL_PATH)
    yield
    application.state.model_bundle = None


app = FastAPI(
    title="Binary Model Service",
    description="Serves predictions from a locally trained model artifact.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    if getattr(request.app.state, "model_bundle", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok", "model_loaded": True}


@app.get("/metadata")
def metadata(request: Request) -> dict[str, Any]:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "target_column": bundle["target_column"],
        "feature_columns": bundle["feature_columns"],
        "numeric_columns": bundle["numeric_columns"],
        "holdout_metrics": bundle.get("metrics", {}),
    }


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request) -> dict[str, Any]:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    expected = bundle["feature_columns"]
    supplied = set(payload.features)
    missing = sorted(set(expected) - supplied)
    extra = sorted(supplied - set(expected))
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    numeric = set(bundle["numeric_columns"])
    values: dict[str, Any] = {}
    for column in expected:
        value = payload.features[column]
        if value is None:
            values[column] = np.nan
        elif column in numeric:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=f"Numeric feature '{column}' must be a finite JSON number or null.",
                )
            if not math.isfinite(float(value)):
                raise HTTPException(
                    status_code=422,
                    detail=f"Numeric feature '{column}' must be finite.",
                )
            values[column] = float(value)
        elif isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Categorical feature '{column}' must be finite.",
                )
            values[column] = str(value)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Categorical feature '{column}' must be a JSON scalar or null.",
            )

    features = pd.DataFrame([values], columns=expected)
    pipeline = bundle["pipeline"]
    prediction = int(pipeline.predict(features)[0])
    raw_probabilities = pipeline.predict_proba(features)[0]
    probabilities = {
        str(int(label)): float(probability)
        for label, probability in zip(pipeline.classes_, raw_probabilities, strict=True)
    }
    return {
        "predicted_label": prediction,
        "probabilities": probabilities,
        "target_column": bundle["target_column"],
    }
