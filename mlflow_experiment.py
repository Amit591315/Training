from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_fscore_support, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_PATH = Path("defects_data.csv")
MODELS_DIR = Path("models")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "defect_prediction_experiment"


def prepare_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["defect_date"])
    df_model = df.copy()

    df_model["month"] = df_model["defect_date"].dt.month
    df_model["quarter"] = df_model["defect_date"].dt.quarter
    df_model["day_of_week"] = df_model["defect_date"].dt.dayofweek

    product_cost = df_model.groupby("product_id")["repair_cost"].agg(["mean", "count"]).reset_index()
    product_cost.columns = ["product_id", "product_avg_cost", "product_defect_count"]
    df_model = df_model.merge(product_cost, on="product_id", how="left")

    le_dict = {}
    categorical_cols = ["defect_type", "defect_location", "severity", "inspection_method"]
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col + "_encoded"] = le.fit_transform(df_model[col])
        le_dict[col] = le

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

    x_train_cost, x_test_cost, y_train_cost, y_test_cost = train_test_split(
        x_cost, y_cost, test_size=0.2, random_state=42
    )
    x_train_sev, x_test_sev, y_train_sev, y_test_sev = train_test_split(
        x_sev, y_sev, test_size=0.2, random_state=42, stratify=y_sev
    )

    return (
        x_train_cost,
        x_test_cost,
        y_train_cost,
        y_test_cost,
        x_train_sev,
        x_test_sev,
        y_train_sev,
        y_test_sev,
        le_dict,
    )


def run_cost_models(x_train, x_test, y_train, y_test):
    models = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "gradient_boosting_regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    best = {"name": None, "model": None, "r2": -np.inf, "rmse": np.inf}

    for name, model in models.items():
        with mlflow.start_run(run_name=f"cost_{name}", nested=True):
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            r2 = float(r2_score(y_test, pred))

            mlflow.log_params({"task": "cost_regression", "model_name": name})
            mlflow.log_metrics({"rmse": rmse, "r2": r2})
            mlflow.sklearn.log_model(model, name="model")

            if r2 > best["r2"]:
                best = {"name": name, "model": model, "r2": r2, "rmse": rmse}

    return best


def run_severity_models(x_train, x_test, y_train, y_test):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest_classifier": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "gradient_boosting_classifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    best = {
        "name": None,
        "model": None,
        "accuracy": -np.inf,
        "precision_weighted": 0.0,
        "recall_weighted": 0.0,
        "f1_weighted": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "f1_macro": 0.0,
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"severity_{name}", nested=True):
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            acc = float(accuracy_score(y_test, pred))
            p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
            p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)

            mlflow.log_params({"task": "severity_classification", "model_name": name})
            mlflow.log_metrics(
                {
                    "accuracy": acc,
                    "precision_weighted": float(p_w),
                    "recall_weighted": float(r_w),
                    "f1_weighted": float(f1_w),
                    "precision_macro": float(p_m),
                    "recall_macro": float(r_m),
                    "f1_macro": float(f1_m),
                }
            )
            mlflow.sklearn.log_model(model, name="model")

            if acc > best["accuracy"]:
                best = {
                    "name": name,
                    "model": model,
                    "accuracy": acc,
                    "precision_weighted": float(p_w),
                    "recall_weighted": float(r_w),
                    "f1_weighted": float(f1_w),
                    "precision_macro": float(p_m),
                    "recall_macro": float(r_m),
                    "f1_macro": float(f1_m),
                }

    return best


def export_best_models(best_cost, best_severity, le_dict):
    MODELS_DIR.mkdir(exist_ok=True)

    cost_path = MODELS_DIR / "best_cost_model.pkl"
    sev_path = MODELS_DIR / "best_sev_model.pkl"
    enc_path = MODELS_DIR / "label_encoders.pkl"

    with open(cost_path, "wb") as f:
        pickle.dump(best_cost["model"], f)

    with open(sev_path, "wb") as f:
        pickle.dump(best_severity["model"], f)

    with open(enc_path, "wb") as f:
        pickle.dump(le_dict, f)

    mlflow.log_artifact(str(cost_path))
    mlflow.log_artifact(str(sev_path))
    mlflow.log_artifact(str(enc_path))


if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="defect_training_pipeline"):
        (
            x_train_cost,
            x_test_cost,
            y_train_cost,
            y_test_cost,
            x_train_sev,
            x_test_sev,
            y_train_sev,
            y_test_sev,
            le_dict,
        ) = prepare_data()

        best_cost = run_cost_models(x_train_cost, x_test_cost, y_train_cost, y_test_cost)
        best_severity = run_severity_models(x_train_sev, x_test_sev, y_train_sev, y_test_sev)

        mlflow.log_params(
            {
                "best_cost_model": best_cost["name"],
                "best_severity_model": best_severity["name"],
            }
        )
        mlflow.log_metrics(
            {
                "best_cost_r2": best_cost["r2"],
                "best_cost_rmse": best_cost["rmse"],
                "best_severity_accuracy": best_severity["accuracy"],
                "best_severity_precision_weighted": best_severity["precision_weighted"],
                "best_severity_recall_weighted": best_severity["recall_weighted"],
                "best_severity_f1_weighted": best_severity["f1_weighted"],
                "best_severity_precision_macro": best_severity["precision_macro"],
                "best_severity_recall_macro": best_severity["recall_macro"],
                "best_severity_f1_macro": best_severity["f1_macro"],
            }
        )

        export_best_models(best_cost, best_severity, le_dict)

    print("MLflow run complete.")
    print(f"Best cost model: {best_cost['name']} | R2={best_cost['r2']:.4f} | RMSE={best_cost['rmse']:.2f}")
    print(
        "Best severity model: "
        f"{best_severity['name']} | "
        f"Accuracy={best_severity['accuracy']:.4f} | "
        f"Precision(weighted)={best_severity['precision_weighted']:.4f} | "
        f"Recall(weighted)={best_severity['recall_weighted']:.4f} | "
        f"F1(weighted)={best_severity['f1_weighted']:.4f}"
    )
    print("Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' and open http://127.0.0.1:5000")
