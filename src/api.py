"""FastAPI inference service for the Adult income model."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/adult_income_pipeline.joblib"))
app = FastAPI(title="Adult Income Prediction API", version="2.0.0")
model: Any | None = None


class AdultFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    age: int = Field(ge=17, le=90)
    workclass: str
    fnlwgt: int = Field(gt=0)
    education: str
    education_num: int = Field(alias="education-num", ge=1, le=16)
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain", ge=0)
    capital_loss: int = Field(alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=99)
    native_country: str = Field(alias="native-country")


def load_model() -> bool:
    global model
    if MODEL_PATH.is_file():
        model = joblib.load(MODEL_PATH)
        return True
    model = None
    return False


@app.on_event("startup")
def startup() -> None:
    load_model()


def _predict_one(payload: AdultFeatures) -> dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model is not available. Train the model first."
        )
    frame = pd.DataFrame([payload.model_dump(by_alias=True)])
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = ">50K" if probability >= 0.5 else "<=50K"
    margin = abs(probability - 0.5)
    confidence = "high" if margin >= 0.3 else "medium" if margin >= 0.15 else "low"
    return {
        "prediction": prediction,
        "probability": probability,
        "confidence": confidence,
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    return {
        "model_name": type(model.named_steps["classifier"]).__name__,
        "features_count": len(model.feature_names_in_)
        if hasattr(model, "feature_names_in_")
        else 14,
    }


@app.post("/predict")
def predict(payload: AdultFeatures) -> dict[str, Any]:
    return _predict_one(payload)


class BatchRequest(BaseModel):
    instances: list[AdultFeatures] = Field(min_length=1, max_length=100)


@app.post("/predict/batch")
def predict_batch(payload: BatchRequest) -> dict[str, list[dict[str, Any]]]:
    return {"predictions": [_predict_one(instance) for instance in payload.instances]}


@app.get("/example")
def example() -> dict[str, Any]:
    return {
        "example_input": {
            "age": 39,
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
        "expected_output": (
            "A predicted income class and model probability, not a guaranteed outcome."
        ),
    }
