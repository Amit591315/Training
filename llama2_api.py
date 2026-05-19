"""
FastAPI service for fine-tuned Llama2 defect prediction models.

Endpoints:
  GET  /health                    — health check
  POST /predict/severity          — predict defect severity
  POST /predict/cost              — predict repair cost
  POST /batch-predict             — batch prediction

Models served via Ollama:
  - llama2-severity-finetuned     — Severity classifier (Minor, Moderate, Critical)
  - llama2-repaircost-finetuned   — Repair cost regressor
"""

import re
import json
import requests
from datetime import datetime
from typing import List, Optional
from urllib.error import URLError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow

# ============================================================================
# Configuration
# ============================================================================

OLLAMA_BASE = "http://127.0.0.1:11434"
SEVERITY_MODEL = "llama2-severity-finetuned"
COST_MODEL = "llama2-repaircost-finetuned"

# Initialize MLflow
mlflow.set_experiment("llama2-inference")

# ============================================================================
# Pydantic Models
# ============================================================================

class DefectInput(BaseModel):
    """Input schema for defect prediction."""
    defect_type: str  # Cosmetic, Functional, Structural
    defect_location: str  # Component, Internal, Surface
    inspection_method: str  # Automated Testing, Manual Testing, Visual Inspection
    product_id: int
    severity: Optional[str] = None  # Optional for cost prediction

class SeverityPrediction(BaseModel):
    """Severity classification result."""
    severity: str  # Minor, Moderate, Critical
    confidence: float
    model: str = SEVERITY_MODEL

class CostPrediction(BaseModel):
    """Repair cost prediction result."""
    predicted_cost: float
    currency: str = "USD"
    model: str = COST_MODEL

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    severity_model: str
    cost_model: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    defects: List[DefectInput]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    results: List[dict]
    total: int
    successful: int
    failed: int

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Llama2 Defect Prediction API",
    description="Fine-tuned Llama2 models for manufacturing defect analysis",
    version="2.0"
)

# ============================================================================
# Helper Functions
# ============================================================================

def call_ollama(model: str, prompt: str, timeout: int = 30) -> str:
    """Call Ollama API and get response."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
            },
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama service unavailable: {str(e)}"
        )

def extract_severity(response: str) -> str:
    """Extract severity from Llama2 response."""
    response_lower = response.lower()
    if "minor" in response_lower:
        return "Minor"
    elif "moderate" in response_lower:
        return "Moderate"
    elif "critical" in response_lower:
        return "Critical"
    else:
        # Default: try to find any capitalized word
        words = response.split()
        for word in words:
            if word.capitalize() in ["Minor", "Moderate", "Critical"]:
                return word.capitalize()
    return "Minor"  # Default fallback

def extract_cost(response: str) -> float:
    """Extract cost from Llama2 response."""
    # Find dollar amounts like $123.45 or just numbers
    import re
    # Look for patterns like $XXX.XX or XXX.XX or $XXX
    patterns = [
        r'\$\s*([\d,]+\.?\d*)',  # $123.45
        r'([\d,]+\.?\d*)\s*(?:dollars?|USD)',  # 123.45 dollars
        r'^\s*([\d,]+\.?\d*)\s*$'  # Just a number
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                cost_str = match.group(1).replace(',', '')
                return float(cost_str)
            except ValueError:
                continue

    # Default fallback
    return 500.0

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        severity_model=SEVERITY_MODEL,
        cost_model=COST_MODEL,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict/severity", response_model=SeverityPrediction)
async def predict_severity(defect: DefectInput):
    """Predict defect severity using fine-tuned Llama2 model."""

    # Construct prompt for Llama2
    prompt = f"""Classify the severity of this manufacturing defect:

Defect Type: {defect.defect_type}
Location: {defect.defect_location}
Inspection Method: {defect.inspection_method}
Product ID: {defect.product_id}

Severity class (Minor, Moderate, or Critical)?"""

    try:
        # Call Ollama
        response = call_ollama(SEVERITY_MODEL, prompt)
        severity = extract_severity(response)

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("model", SEVERITY_MODEL)
            mlflow.log_param("defect_type", defect.defect_type)
            mlflow.log_param("defect_location", defect.defect_location)
            mlflow.log_metric("prediction_made", 1)
            mlflow.log_dict({
                "input": defect.dict(),
                "output": severity,
                "raw_response": response
            }, "severity_prediction.json")

        return SeverityPrediction(
            severity=severity,
            confidence=0.85  # Default confidence for Llama2
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/cost", response_model=CostPrediction)
async def predict_cost(defect: DefectInput):
    """Predict repair cost using fine-tuned Llama2 model."""

    # Construct prompt for Llama2
    prompt = f"""Estimate the repair cost for this manufacturing defect:

Defect Type: {defect.defect_type}
Location: {defect.defect_location}
Inspection Method: {defect.inspection_method}
Product ID: {defect.product_id}
Severity: {defect.severity if defect.severity else 'Unknown'}

Estimated repair cost in USD?"""

    try:
        # Call Ollama
        response = call_ollama(COST_MODEL, prompt)
        cost = extract_cost(response)

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("model", COST_MODEL)
            mlflow.log_param("defect_type", defect.defect_type)
            mlflow.log_param("defect_location", defect.defect_location)
            mlflow.log_metric("predicted_cost", cost)
            mlflow.log_dict({
                "input": defect.dict(),
                "predicted_cost": cost,
                "raw_response": response
            }, "cost_prediction.json")

        return CostPrediction(predicted_cost=cost)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction for multiple defects."""
    results = []
    successful = 0
    failed = 0

    for idx, defect in enumerate(request.defects):
        try:
            # Predict severity
            severity_result = await predict_severity(defect)

            # Predict cost
            cost_result = await predict_cost(defect)

            results.append({
                "id": idx,
                "defect": defect.dict(),
                "severity": severity_result.dict(),
                "cost": cost_result.dict()
            })
            successful += 1

        except Exception as e:
            results.append({
                "id": idx,
                "error": str(e)
            })
            failed += 1

    return BatchPredictionResponse(
        results=results,
        total=len(request.defects),
        successful=successful,
        failed=failed
    )

@app.post("/explain")
async def explain_prediction(defect: DefectInput):
    """Get detailed explanation of predictions."""
    return {
        "input": defect.dict(),
        "severity_explanation": "Based on defect type, location, and inspection method",
        "cost_explanation": "Based on defect severity and type",
        "models": {
            "severity": SEVERITY_MODEL,
            "cost": COST_MODEL
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
