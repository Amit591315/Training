"""Compatibility FastAPI service for classic (non-Llama2) defect prediction.

This module is kept to support existing CI and tests that expect `api:app`.
It loads pre-trained sklearn models from `models/` when available.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "defects_data.csv"
MODELS_DIR = BASE_DIR / "models"
COST_MODEL_PATH = MODELS_DIR / "best_cost_model.pkl"
SEV_MODEL_PATH = MODELS_DIR / "best_sev_model.pkl"
ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"


app = FastAPI(
    title="Defect Prediction API",
    version="1.0",
    description="Classic sklearn-backed API kept for CI compatibility.",
)


class CostPredictionRequest(BaseModel):
    defect_type: Literal["Cosmetic", "Functional", "Structural"]
    defect_location: Literal["Component", "Internal", "Surface"]
    severity: Literal["Minor", "Moderate", "Critical"]
    inspection_method: Literal["Visual Inspection", "Manual Testing", "Automated Testing"]
    month: int = Field(ge=1, le=12)
    quarter: int = Field(ge=1, le=4)
    day_of_week: int = Field(ge=0, le=6)
    product_avg_cost: float = Field(ge=0)
    product_defect_count: int = Field(ge=0)


class SeverityPredictionRequest(BaseModel):
    defect_type: Literal["Cosmetic", "Functional", "Structural"]
    defect_location: Literal["Component", "Internal", "Surface"]
    inspection_method: Literal["Visual Inspection", "Manual Testing", "Automated Testing"]
    product_id: int = Field(ge=0)
    month: int = Field(ge=1, le=12)
    quarter: int = Field(ge=1, le=4)
    day_of_week: int = Field(ge=0, le=6)
    product_avg_cost: float = Field(ge=0)
    product_defect_count: int = Field(ge=0)


cost_model = None
sev_model = None
label_encoders: Dict[str, LabelEncoder] = {}


def _train_fallback_models() -> tuple[object, object, Dict[str, LabelEncoder]]:
    if not DATA_PATH.exists():
        raise RuntimeError("No saved models or training dataset available.")

    df = pd.read_csv(DATA_PATH, parse_dates=["defect_date"])
    df_model = df.copy()

    df_model["month"] = df_model["defect_date"].dt.month
    df_model["quarter"] = df_model["defect_date"].dt.quarter
    df_model["day_of_week"] = df_model["defect_date"].dt.dayofweek

    product_cost = df_model.groupby("product_id")["repair_cost"].agg(["mean", "count"]).reset_index()
    product_cost.columns = ["product_id", "product_avg_cost", "product_defect_count"]
    df_model = df_model.merge(product_cost, on="product_id", how="left")

    encoders: Dict[str, LabelEncoder] = {}
    categorical_cols = ["defect_type", "defect_location", "severity", "inspection_method"]
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_model[f"{col}_encoded"] = encoder.fit_transform(df_model[col])
        encoders[col] = encoder

    x_cost = df_model[
        [
            "defect_type_encoded",
            "defect_location_encoded",
            "severity_encoded",
            "inspection_method_encoded",
            "month",
            "quarter",
            "day_of_week",
            "product_avg_cost",
            "product_defect_count",
        ]
    ]
    y_cost = df_model["repair_cost"]

    x_sev = df_model[
        [
            "defect_type_encoded",
            "defect_location_encoded",
            "inspection_method_encoded",
            "product_id",
            "month",
            "quarter",
            "day_of_week",
            "product_avg_cost",
            "product_defect_count",
        ]
    ]
    y_sev = df_model["severity_encoded"]

    x_train_cost, _, y_train_cost, _ = train_test_split(x_cost, y_cost, test_size=0.2, random_state=42)
    x_train_sev, _, y_train_sev, _ = train_test_split(
        x_sev,
        y_sev,
        test_size=0.2,
        random_state=42,
        stratify=y_sev,
    )

    fallback_cost = LinearRegression()
    fallback_cost.fit(x_train_cost, y_train_cost)

    fallback_sev = LogisticRegression(max_iter=1000, random_state=42)
    fallback_sev.fit(x_train_sev, y_train_sev)

    return fallback_cost, fallback_sev, encoders


def _load_or_prepare_models() -> None:
    global cost_model, sev_model, label_encoders

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if COST_MODEL_PATH.exists() and SEV_MODEL_PATH.exists() and ENCODERS_PATH.exists():
        with open(COST_MODEL_PATH, "rb") as f:
            cost_model = pickle.load(f)
        with open(SEV_MODEL_PATH, "rb") as f:
            sev_model = pickle.load(f)
        with open(ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
        return

    cost_model, sev_model, label_encoders = _train_fallback_models()

    with open(COST_MODEL_PATH, "wb") as f:
        pickle.dump(cost_model, f)
    with open(SEV_MODEL_PATH, "wb") as f:
        pickle.dump(sev_model, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(label_encoders, f)


def _ensure_models_loaded() -> None:
    if cost_model is None or sev_model is None or not label_encoders:
        _load_or_prepare_models()


def _encode(col: str, value: str) -> int:
    encoder = label_encoders[col]
    try:
        return int(encoder.transform([value])[0])
    except Exception as exc:  # pragma: no cover - protected by pydantic validation
        raise HTTPException(status_code=422, detail=f"Invalid {col}: {value}") from exc


@app.on_event("startup")
def startup() -> None:
    _load_or_prepare_models()


@app.get("/health")
def health() -> dict:
    _ensure_models_loaded()
    return {
        "status": "ok",
        "cost_model": type(cost_model).__name__ if cost_model is not None else "unloaded",
        "sev_model": type(sev_model).__name__ if sev_model is not None else "unloaded",
    }


@app.post("/predict/cost")
def predict_cost(payload: CostPredictionRequest) -> dict:
    try:
        _ensure_models_loaded()
        features = np.array(
            [
                [
                    _encode("defect_type", payload.defect_type),
                    _encode("defect_location", payload.defect_location),
                    _encode("severity", payload.severity),
                    _encode("inspection_method", payload.inspection_method),
                    payload.month,
                    payload.quarter,
                    payload.day_of_week,
                    payload.product_avg_cost,
                    payload.product_defect_count,
                ]
            ]
        )
        predicted_cost = float(cost_model.predict(features)[0])
        return {"predicted_repair_cost": max(0.0, predicted_cost)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cost prediction failed: {exc}") from exc


@app.post("/predict/severity")
def predict_severity(payload: SeverityPredictionRequest) -> dict:
    try:
        _ensure_models_loaded()
        features = np.array(
            [
                [
                    _encode("defect_type", payload.defect_type),
                    _encode("defect_location", payload.defect_location),
                    _encode("inspection_method", payload.inspection_method),
                    payload.product_id,
                    payload.month,
                    payload.quarter,
                    payload.day_of_week,
                    payload.product_avg_cost,
                    payload.product_defect_count,
                ]
            ]
        )

        pred_label = int(sev_model.predict(features)[0])
        severity = str(label_encoders["severity"].inverse_transform([pred_label])[0])

        probabilities = None
        if hasattr(sev_model, "predict_proba"):
            probs = sev_model.predict_proba(features)[0]
            class_labels = label_encoders["severity"].inverse_transform(sev_model.classes_)
            probabilities = {str(label): float(prob) for label, prob in zip(class_labels, probs)}

        response = {"predicted_severity": severity}
        if probabilities is not None:
            response["probabilities"] = probabilities
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Severity prediction failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
