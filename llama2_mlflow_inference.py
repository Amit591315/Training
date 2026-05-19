"""
MLflow tracking for Llama2 inference metrics and performance monitoring.
Logs predictions, latency, and model performance to MLflow.
"""

import time
import json
import pandas as pd
import mlflow
from datetime import datetime
import requests

# Configuration
OLLAMA_API = "http://localhost:11434"
SEVERITY_MODEL = "llama2-severity-finetuned"
COST_MODEL = "llama2-repaircost-finetuned"

class Llama2InferenceTracker:
    """Track Llama2 inference metrics to MLflow."""

    def __init__(self, experiment_name="llama2-inference"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.predictions = []

    def predict_severity(self, defect_input: dict) -> dict:
        """Predict severity and track metrics."""
        start_time = time.time()

        try:
            # Make prediction
            response = requests.post(
                "http://localhost:8001/predict/severity",
                json=defect_input,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            latency = time.time() - start_time

            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_param("task", "severity_classification")
                mlflow.log_param("model", SEVERITY_MODEL)
                mlflow.log_param("defect_type", defect_input.get("defect_type"))
                mlflow.log_param("defect_location", defect_input.get("defect_location"))

                mlflow.log_metric("inference_latency_s", latency)
                mlflow.log_metric("confidence", result.get("confidence", 0.85))
                mlflow.log_dict({
                    "input": defect_input,
                    "output": result,
                    "latency_seconds": latency,
                    "timestamp": datetime.now().isoformat()
                }, "severity_prediction.json")

            self.predictions.append({
                "task": "severity",
                "input": defect_input,
                "output": result,
                "latency": latency,
                "timestamp": datetime.now()
            })

            return result

        except Exception as e:
            print(f"Error predicting severity: {e}")
            return None

    def predict_cost(self, defect_input: dict) -> dict:
        """Predict cost and track metrics."""
        start_time = time.time()

        try:
            # Make prediction
            response = requests.post(
                "http://localhost:8001/predict/cost",
                json=defect_input,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            latency = time.time() - start_time

            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_param("task", "cost_regression")
                mlflow.log_param("model", COST_MODEL)
                mlflow.log_param("defect_type", defect_input.get("defect_type"))
                mlflow.log_param("defect_location", defect_input.get("defect_location"))

                mlflow.log_metric("inference_latency_s", latency)
                mlflow.log_metric("predicted_cost", result.get("predicted_cost", 0))
                mlflow.log_dict({
                    "input": defect_input,
                    "output": result,
                    "latency_seconds": latency,
                    "timestamp": datetime.now().isoformat()
                }, "cost_prediction.json")

            self.predictions.append({
                "task": "cost",
                "input": defect_input,
                "output": result,
                "latency": latency,
                "timestamp": datetime.now()
            })

            return result

        except Exception as e:
            print(f"Error predicting cost: {e}")
            return None

    def batch_predict_from_csv(self, csv_path: str, limit: int = None):
        """Run batch predictions on data from CSV and track metrics."""
        print(f"\n=== Batch Inference on {csv_path} ===\n")

        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)

        results = []

        with mlflow.start_run():
            mlflow.log_param("task", "batch_inference")
            mlflow.log_param("data_source", csv_path)
            mlflow.log_metric("total_samples", len(df))

            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    print(f"Processing {idx}/{len(df)}...")

                defect_input = {
                    "defect_type": row["defect_type"],
                    "defect_location": row["defect_location"],
                    "inspection_method": row["inspection_method"],
                    "product_id": int(row["product_id"])
                }

                # Predict severity
                severity_result = self.predict_severity(defect_input)

                # Add severity for cost prediction
                defect_input["severity"] = severity_result.get("severity") if severity_result else "Unknown"

                # Predict cost
                cost_result = self.predict_cost(defect_input)

                results.append({
                    "defect_type": row["defect_type"],
                    "actual_severity": row["severity"],
                    "predicted_severity": severity_result.get("severity") if severity_result else "N/A",
                    "actual_cost": row["repair_cost"],
                    "predicted_cost": cost_result.get("predicted_cost") if cost_result else None
                })

            # Log batch summary
            results_df = pd.DataFrame(results)

            # Calculate accuracy for severity (if applicable)
            if "predicted_severity" in results_df.columns:
                accuracy = (results_df["actual_severity"] == results_df["predicted_severity"]).sum() / len(results_df)
                mlflow.log_metric("severity_accuracy", accuracy)
                print(f"Severity Accuracy: {accuracy:.2%}")

            # Calculate RMSE for cost
            if "predicted_cost" in results_df.columns:
                cost_rmse = ((results_df["actual_cost"] - results_df["predicted_cost"]) ** 2).mean() ** 0.5
                mlflow.log_metric("cost_rmse", cost_rmse)
                print(f"Cost RMSE: ${cost_rmse:.2f}")

                cost_mae = abs(results_df["actual_cost"] - results_df["predicted_cost"]).mean()
                mlflow.log_metric("cost_mae", cost_mae)
                print(f"Cost MAE: ${cost_mae:.2f}")

            # Save results
            results_df.to_csv("llama2_batch_predictions.csv", index=False)
            mlflow.log_artifact("llama2_batch_predictions.csv")

        return results_df

def main():
    print("=" * 70)
    print("Llama2 Inference MLflow Tracking")
    print("=" * 70)

    # Initialize tracker
    tracker = Llama2InferenceTracker()

    # Example predictions
    print("\n=== Single Predictions ===\n")

    test_defect_1 = {
        "defect_type": "Structural",
        "defect_location": "Internal",
        "inspection_method": "Automated Testing",
        "product_id": 42
    }

    print("1. Predicting severity...")
    severity = tracker.predict_severity(test_defect_1)
    print(f"   Result: {severity}")

    test_defect_1["severity"] = severity.get("severity") if severity else "Unknown"

    print("\n2. Predicting cost...")
    cost = tracker.predict_cost(test_defect_1)
    print(f"   Result: {cost}")

    # Batch predictions
    print("\n\n=== Batch Predictions ===")
    results = tracker.batch_predict_from_csv("defects_data.csv", limit=50)

    print("\n" + "=" * 70)
    print("✓ Inference tracking complete!")
    print(f"Results saved to: llama2_batch_predictions.csv")
    print("=" * 70)

if __name__ == "__main__":
    main()
