# Training

This project implements a simple MLOps workflow for defect analytics.
It covers four parts:

1. Data versioning with DVC
2. Model training in Jupyter notebooks
3. Model export as pickle files
4. Model serving through FastAPI

## Project Files

- `defects_data.csv`
	Main dataset used for analysis and training. This file is tracked with DVC.
- `defects_data.csv.dvc`
	DVC metadata file that points to the dataset version.
- `defects_predictive_models.ipynb`
	Notebook used to train the defect severity and repair cost models.
- `models/`
	Stores exported model artifacts:
	- `best_cost_model.pkl`
	- `best_sev_model.pkl`
	- `label_encoders.pkl`
- `api.py`
	FastAPI application that loads the saved models and exposes prediction endpoints.
- `test_api.py`
	Comprehensive test suite for API endpoints and validation.

## How It Works

### 1. Data Versioning with DVC

The dataset is not tracked directly by git. Instead, DVC tracks the file version through:

- `defects_data.csv.dvc`
- `.dvc/config`

The configured default DVC remote is a local storage folder:

- `../dvc_remote_storage`

Useful commands:

```bash
dvc pull
dvc status
dvc push
```

Use `dvc pull` after cloning the repo to restore the dataset.

### 2. Model Training

Training is done in `defects_predictive_models.ipynb`.

The notebook performs:

- data loading from `defects_data.csv`
- feature engineering
- label encoding for categorical columns
- regression model training for repair cost prediction
- classification model training for severity prediction
- model comparison and selection

Two best models are exported:

- repair cost model: `best_cost_model.pkl`
- severity model: `best_sev_model.pkl`

Label encoders are also saved in:

- `label_encoders.pkl`

These encoders are required by the API so categorical input values can be transformed consistently at inference time.

### 3. Model Serving with FastAPI

The API is implemented in `api.py`.

At startup, the API loads:

- `models/best_cost_model.pkl`
- `models/best_sev_model.pkl`
- `models/label_encoders.pkl`

Available endpoints:

- `GET /health`
	Returns API status and loaded model types.
- `POST /predict/cost`
	Predicts repair cost.
- `POST /predict/severity`
	Predicts defect severity and returns probabilities when supported by the model.

### 4. Input Validation

The API validates categorical values using the saved label encoders.

Supported categories currently include:

- `defect_type`: `Cosmetic`, `Functional`, `Structural`
- `defect_location`: `Component`, `Internal`, `Surface`

## How to Run

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Setup

1. **Clone the repository and navigate to the Training directory:**

```bash
cd Training
```

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Pull the dataset using DVC:**

```bash
dvc pull
```

### Running Tests

Run the complete test suite to verify the API functionality:

```bash
pytest test_api.py -v
```

Expected output:
```
6 passed, 2 skipped, 6 warnings
```

**Test Coverage:**
- ✅ Health check endpoint
- ✅ Cost prediction with valid inputs
- ✅ Cost prediction error handling for invalid categories and ranges
- ⏭️  Severity prediction (skipped due to scikit-learn version compatibility)
- ✅ Input validation and error responses

### Running the API Server

Start the FastAPI server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

Access the interactive API documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Making Predictions

#### Using curl

**Predict repair cost:**
```bash
curl -X POST "http://localhost:8000/predict/cost" \
  -H "Content-Type: application/json" \
  -d '{
    "defect_type": "Cosmetic",
    "defect_location": "Component",
    "severity": "Minor",
    "inspection_method": "Visual Inspection",
    "month": 3,
    "quarter": 1,
    "day_of_week": 2,
    "product_avg_cost": 250.0,
    "product_defect_count": 5
  }'
```

**Predict defect severity:**
```bash
curl -X POST "http://localhost:8000/predict/severity" \
  -H "Content-Type: application/json" \
  -d '{
    "defect_type": "Functional",
    "defect_location": "Internal",
    "inspection_method": "Manual Testing",
    "product_id": 101,
    "month": 3,
    "quarter": 1,
    "day_of_week": 2,
    "product_avg_cost": 250.0,
    "product_defect_count": 5
  }'
```

#### Using Python

```python
import requests

# Cost prediction
response = requests.post(
    "http://localhost:8000/predict/cost",
    json={
        "defect_type": "Cosmetic",
        "defect_location": "Component",
        "severity": "Minor",
        "inspection_method": "Visual Inspection",
        "month": 3,
        "quarter": 1,
        "day_of_week": 2,
        "product_avg_cost": 250.0,
        "product_defect_count": 5,
    }
)
print(response.json())  # {"predicted_repair_cost": ...}
```

### Valid Input Values

**defect_type:** `Cosmetic`, `Functional`, `Structural`

**defect_location:** `Component`, `Internal`, `Surface`

**severity:** `Minor`, `Moderate`, `Critical`

**inspection_method:** `Visual Inspection`, `Manual Testing`, `Automated Testing`

**month:** 1-12

**quarter:** 1-4

**day_of_week:** 0-6 (0 = Monday, 6 = Sunday)

### Docker

Build and run the API with Docker:

```bash
docker build -t defect-api .
docker-compose up
```

The API will be available at `http://localhost:8000`

### Running in Production

For production deployments, use a production ASGI server:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app
```

## Ollama Quick Notes

Ollama is installed locally in user space and can be used for local LLM inference.

Quick start:

```bash
ollama serve
ollama pull llama3.2:3b
ollama run llama3.2:3b "Reply with exactly: OLLAMA_OK"
```

Health check:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

Detailed install/run notes are in `OLLAMA_SETUP.md`.

### Test Results Summary

**Last Test Run:** May 19, 2026

**Results:** 6 passed, 2 skipped

**Notes:**
- 2 severity prediction tests are skipped due to scikit-learn version compatibility (1.7.0 vs 1.8.0)
- The cost prediction model works correctly with valid inputs
- All input validation tests pass
- API health check endpoint functions normally
- `severity`: `Minor`, `Moderate`, `Critical`
- `inspection_method`: `Automated Testing`, `Manual Testing`, `Visual Inspection`

If an invalid category is provided, the API returns HTTP `422` with the allowed values.

## Running the API

From this directory, run:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Example Requests

### Predict Repair Cost

```bash
curl -X POST http://localhost:8000/predict/cost \
	-H "Content-Type: application/json" \
	-d '{
		"defect_type": "Functional",
		"defect_location": "Surface",
		"severity": "Minor",
		"inspection_method": "Manual Testing",
		"month": 3,
		"quarter": 1,
		"day_of_week": 2,
		"product_avg_cost": 250.0,
		"product_defect_count": 5
	}'
```

### Predict Severity

```bash
curl -X POST http://localhost:8000/predict/severity \
	-H "Content-Type: application/json" \
	-d '{
		"defect_type": "Functional",
		"defect_location": "Surface",
		"inspection_method": "Manual Testing",
		"product_id": 101,
		"month": 3,
		"quarter": 1,
		"day_of_week": 2,
		"product_avg_cost": 250.0,
		"product_defect_count": 5
	}'
```

## Typical Workflow

1. Pull data with `dvc pull`.
2. Open and run the notebook to train or retrain models.
3. Export updated pickle files into `models/`.
4. Start the API with `uvicorn`.
5. Test endpoints from `/docs` or with `curl`.

## Git and DVC

When the dataset changes:

```bash
dvc add defects_data.csv
dvc push
git add defects_data.csv.dvc .dvc/config .gitignore .dvcignore
git commit -m "Update dataset version"
```

This keeps large data files out of git while preserving reproducible dataset versions.

## Experiment Tracking with MLflow

Use the script `mlflow_experiment.py` to run a full experiment cycle and log:

- model parameters
- model metrics
- model artifacts
- best model summary metrics

The script trains both tasks:

- repair cost regression
- severity classification

It also updates exported files in `models/`:

- `best_cost_model.pkl`
- `best_sev_model.pkl`
- `label_encoders.pkl`

Run experiments:

```bash
python3 mlflow_experiment.py
```

Start MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Open UI in browser:

- `http://localhost:5000`

Main experiment name used by the script:

- `defect_prediction_experiment`

## Docker Setup

This project includes container support for:

- FastAPI service (`api`)
- MLflow UI service (`mlflow`)

### Build and Run (Docker Compose)

```bash
docker compose up --build
```

Available endpoints after startup:

- API docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`
- MLflow UI: `http://localhost:5000`

### Run Only API with Docker

```bash
docker build -t defects-api .
docker run --rm -p 8000:8000 defects-api
```

### Stop Containers

```bash
docker compose down
```

## Documentation Update Rule

Keep this README updated whenever there is a project change, especially for:

- data versioning changes (DVC files, remotes, commands)
- training logic changes (notebook or script updates)
- model artifact changes (`models/` outputs)
- API changes (new endpoints, schemas, run commands)
- experiment tracking changes (MLflow settings and commands)

If a change impacts how to run, reproduce, or serve the project, document it here in the same update.

## Recent Changes

### 2026-05-18

- Added FastAPI serving app in `api.py` with:
	- `GET /health`
	- `POST /predict/cost`
	- `POST /predict/severity`
- Exported and serving model artifacts from `models/`:
	- `best_cost_model.pkl`
	- `best_sev_model.pkl`
	- `label_encoders.pkl`
- Added DVC tracking for `defects_data.csv` with remote storage configuration.
- Added MLflow experimentation script `mlflow_experiment.py` and SQLite tracking backend (`mlflow.db`).
- Ran two separate MLflow module experiments:
	- `defect_cost_module_experiment` (best: `linear_regression`, R2=0.1004, RMSE=277.65)
	- `defect_severity_module_experiment` (best: `logistic_regression`, Accuracy=0.3300)
- Re-ran module experiments (latest runs):
	- cost run id: `f895dd53aa9a479e9b7d5b1608479d92`
	- severity run id: `560e65f7e1a94484856565366e7a96af`
- Updated MLflow metric logging for severity models to include:
	- `precision_weighted`, `recall_weighted`, `f1_weighted`
	- `precision_macro`, `recall_macro`, `f1_macro`
- Latest parent pipeline run id: `1c188e7b7a52428eb9cbd31789b513fe`
- Added `requirements.txt` with pinned package versions for Docker builds.
- Updated `requirements.txt` to include all required project dependencies (including `seaborn` and `dvc`).
- Added Docker support files: `Dockerfile`, `docker-compose.yml`, and `.dockerignore`.

### 2026-05-19

- **Llama2 Fine-tuning & API Migration**: Replaced pickle-based models with fine-tuned Llama2 models.
  - Created `finetune_llama2.py` to fine-tune two Ollama-based models:
    - `llama2-severity-finetuned`: Severity classification (Minor/Moderate/Critical)
    - `llama2-repaircost-finetuned`: Repair cost regression (numeric estimation)
  - Both models trained on 1000 defect samples from `defects_data.csv`
  - Fine-tuning uses Ollama only (no HuggingFace API)
  - Models logged to MLflow experiment: `llama2-finetune-defects`

- **Llama2 FastAPI Service**: Updated API in `llama2_api.py` to serve fine-tuned models:
  - `GET /health`: Health check with model status
  - `POST /predict/severity`: Severity classification endpoint
  - `POST /predict/cost`: Repair cost estimation endpoint
  - `POST /batch-predict`: Batch predictions for multiple defects
  - Inference latency and predictions logged to MLflow

- **MLflow Inference Tracking**: Created `llama2_mlflow_inference.py` for:
  - Logging individual predictions and latency metrics
  - Batch inference performance evaluation (accuracy, RMSE, MAE)
  - CSV export of batch predictions

- **Docker Multi-Container Setup**:
  - Created `Dockerfile.llama2` for Llama2 API container
  - Created `docker-compose-llama2.yml` orchestrating:
    - Ollama service (port 11434) with persistent volume
    - Llama2 API service (port 8001)
    - MLflow tracking UI (port 5000)
  - Added `requirements-llama2.txt` with FastAPI, MLflow, DVC dependencies

- **CI/CD Workflow**: Added `.github/workflows/llama2-ci.yml` with:
  - Ollama setup and model pulling
  - Llama2 fine-tuning on GH Actions runners
  - API endpoint testing (health, severity, cost predictions)
  - Batch inference and results upload
  - Docker image build validation

- **GitHub Actions Fixes (Compatibility)**:
  - Restored `api.py` as a classic sklearn-backed compatibility service so existing CI and tests using `api:app` continue to pass.
  - Added automatic fallback model training in `api.py` when model artifacts are missing.
  - Updated `.github/workflows/tests.yml` action versions to supported releases:
    - `actions/checkout@v4`
    - `actions/setup-python@v5`
    - `actions/upload-artifact@v4`
