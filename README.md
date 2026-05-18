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
