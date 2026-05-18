"""
Defect Prediction API
  POST /predict/cost      — Predict repair cost (regression)
  POST /predict/severity  — Predict defect severity (classification)
  GET  /health            — Health check
"""

import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"


def _load(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        raise RuntimeError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


cost_model = _load("best_cost_model.pkl")
sev_model = _load("best_sev_model.pkl")
le_dict = _load("label_encoders.pkl")

SEVERITY_LABELS = {0: "Minor", 1: "Moderate", 2: "Critical"}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CostRequest(BaseModel):
    """Input for repair-cost prediction."""

    defect_type: str = Field(..., examples=["Crack"])
    defect_location: str = Field(..., examples=["Engine"])
    severity: str = Field(..., examples=["Minor"])
    inspection_method: str = Field(..., examples=["Visual"])
    month: int = Field(..., ge=1, le=12, examples=[3])
    quarter: int = Field(..., ge=1, le=4, examples=[1])
    day_of_week: int = Field(..., ge=0, le=6, examples=[2])
    product_avg_cost: float = Field(..., ge=0, examples=[250.0])
    product_defect_count: int = Field(..., ge=0, examples=[5])


class CostResponse(BaseModel):
    predicted_repair_cost: float


class SeverityRequest(BaseModel):
    """Input for defect-severity classification."""

    defect_type: str = Field(..., examples=["Crack"])
    defect_location: str = Field(..., examples=["Engine"])
    inspection_method: str = Field(..., examples=["Visual"])
    product_id: int = Field(..., ge=0, examples=[101])
    month: int = Field(..., ge=1, le=12, examples=[3])
    quarter: int = Field(..., ge=1, le=4, examples=[1])
    day_of_week: int = Field(..., ge=0, le=6, examples=[2])
    product_avg_cost: float = Field(..., ge=0, examples=[250.0])
    product_defect_count: int = Field(..., ge=0, examples=[5])


class SeverityResponse(BaseModel):
    predicted_severity: str
    predicted_severity_code: int
    probabilities: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Defect Prediction API",
    description="Predict repair cost and defect severity from defect attributes.",
    version="1.0.0",
)


def _encode(col: str, value: str) -> int:
    """Label-encode a single categorical value."""
    encoder = le_dict.get(col)
    if encoder is None:
        raise HTTPException(status_code=500, detail=f"No encoder found for '{col}'")
    if value not in encoder.classes_:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown value '{value}' for '{col}'. "
            f"Valid values: {list(encoder.classes_)}",
        )
    return int(encoder.transform([value])[0])


@app.get("/health")
def health():
    return {"status": "ok", "cost_model": type(cost_model).__name__, "sev_model": type(sev_model).__name__}


@app.post("/predict/cost", response_model=CostResponse)
def predict_cost(req: CostRequest):
    """Predict repair cost for a given defect."""
    features = np.array([[
        _encode("defect_type", req.defect_type),
        _encode("defect_location", req.defect_location),
        _encode("severity", req.severity),
        _encode("inspection_method", req.inspection_method),
        req.month,
        req.quarter,
        req.day_of_week,
        req.product_avg_cost,
        req.product_defect_count,
    ]])
    prediction = float(cost_model.predict(features)[0])
    return CostResponse(predicted_repair_cost=round(prediction, 2))


@app.post("/predict/severity", response_model=SeverityResponse)
def predict_severity(req: SeverityRequest):
    """Predict defect severity (Minor / Moderate / Critical)."""
    features = np.array([[
        _encode("defect_type", req.defect_type),
        _encode("defect_location", req.defect_location),
        _encode("inspection_method", req.inspection_method),
        req.product_id,
        req.month,
        req.quarter,
        req.day_of_week,
        req.product_avg_cost,
        req.product_defect_count,
    ]])
    code = int(sev_model.predict(features)[0])
    label = SEVERITY_LABELS.get(code, str(code))

    probs = None
    if hasattr(sev_model, "predict_proba"):
        raw = sev_model.predict_proba(features)[0]
        probs = {SEVERITY_LABELS.get(i, str(i)): round(float(p), 4) for i, p in enumerate(raw)}

    return SeverityResponse(
        predicted_severity=label,
        predicted_severity_code=code,
        probabilities=probs,
    )
