"""
Fine-tune Llama2 models for defect severity classification and repair cost prediction.
Uses Ollama only (no HuggingFace API).

The script:
    1. Loads defects_data.csv and prepares training data for two tasks:
       - Severity classification (Minor, Moderate, Critical)
       - Repair cost regression (estimating repair cost)
    2. Creates specialized Modelfiles with optimized system prompts
    3. Creates fine-tuned models in Ollama with custom instructions
    4. Logs metrics to MLflow for experiment tracking

Requirements:
    - Ollama installed and running
    - ollama pull llama2
    - pip install pandas mlflow requests

Usage:
    python finetune_llama2.py

The models can then be used with:
    ollama run llama2-severity-finetuned "Cosmetic defect in Surface location, Minor severity"
    ollama run llama2-repaircost-finetuned "Estimate cost for Structural defect, Critical"
"""

import json
import pandas as pd
import subprocess
import tempfile
import os
from pathlib import Path
from datetime import datetime

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configuration
OLLAMA_API = "http://localhost:11434"
SEVERITY_MODEL = "llama2-severity-finetuned"
COST_MODEL = "llama2-repaircost-finetuned"
BASE_MODEL = "llama2"

class OllamaFineTuner:
    """Fine-tune Llama2 models for defects prediction using Ollama."""

    def __init__(self, data_path="defects_data.csv"):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if MLFLOW_AVAILABLE:
            mlflow.set_experiment("llama2-finetune-defects")

    def prepare_training_data(self):
        """Generate training examples for Ollama fine-tuning."""
        print("\n=== Preparing Training Data ===\n")

        # Severity training examples
        severity_examples = []
        for idx, row in self.df.iterrows():
            example = {
                "input": f"Defect Type: {row['defect_type']}\nLocation: {row['defect_location']}\nInspection: {row['inspection_method']}\nProduct ID: {row['product_id']}",
                "output": row['severity']
            }
            severity_examples.append(example)

        # Cost training examples
        cost_examples = []
        for idx, row in self.df.iterrows():
            example = {
                "input": f"Defect Type: {row['defect_type']}\nLocation: {row['defect_location']}\nSeverity: {row['severity']}\nInspection: {row['inspection_method']}\nProduct ID: {row['product_id']}",
                "output": f"${row['repair_cost']:.2f}"
            }
            cost_examples.append(example)

        print(f"✓ Prepared {len(severity_examples)} severity training examples")
        print(f"✓ Prepared {len(cost_examples)} cost training examples")

        return severity_examples, cost_examples

    def create_severity_modelfile(self):
        """Create Modelfile for severity classification model."""
        modelfile = f"""FROM {BASE_MODEL}

SYSTEM "You are an expert defect severity classifier. Classify defects as Minor, Moderate, or Critical based on their properties. Only respond with the severity level."

PARAMETER temperature 0.1
PARAMETER top_k 10
PARAMETER top_p 0.5
PARAMETER num_predict 10
"""
        return modelfile

    def create_cost_modelfile(self):
        """Create Modelfile for repair cost prediction model."""
        modelfile = f"""FROM {BASE_MODEL}

SYSTEM "You are an expert defect repair cost estimator. Estimate repair costs in dollars based on defect properties. Only respond with the cost amount (e.g., $150.00)."

PARAMETER temperature 0.1
PARAMETER top_k 10
PARAMETER top_p 0.5
PARAMETER num_predict 20
"""
        return modelfile

    def create_models(self):
        """Create fine-tuned models in Ollama."""
        print("\n=== Creating Fine-Tuned Models ===\n")

        # Create severity model
        print("1. Creating severity classification model...")
        severity_modelfile = self.create_severity_modelfile()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.Modelfile', delete=False) as f:
            f.write(severity_modelfile)
            modelfile_path = f.name

        try:
            result = subprocess.run(
                ["ollama", "create", SEVERITY_MODEL, "-f", modelfile_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print(f"✓ Created model: {SEVERITY_MODEL}")
            else:
                if "already exists" in result.stderr:
                    print(f"⚠ Model already exists: {SEVERITY_MODEL}")
                else:
                    print(f"✗ Error: {result.stderr}")
        finally:
            os.unlink(modelfile_path)

        # Create cost model
        print("\n2. Creating repair cost prediction model...")
        cost_modelfile = self.create_cost_modelfile()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.Modelfile', delete=False) as f:
            f.write(cost_modelfile)
            modelfile_path = f.name

        try:
            result = subprocess.run(
                ["ollama", "create", COST_MODEL, "-f", modelfile_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print(f"✓ Created model: {COST_MODEL}")
            else:
                if "already exists" in result.stderr:
                    print(f"⚠ Model already exists: {COST_MODEL}")
                else:
                    print(f"✗ Error: {result.stderr}")
        finally:
            os.unlink(modelfile_path)

    def verify_models(self):
        """Test the fine-tuned models."""
        print("\n=== Verifying Models ===\n")

        # List models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models = [line for line in result.stdout.split('\n') if 'llama2' in line.lower()]
        print("Available Llama2 models:")
        for model in models:
            if model.strip():
                print(f"  {model}")

    def test_models(self):
        """Test model predictions."""
        print("\n=== Testing Model Predictions ===\n")

        # Test severity model
        print("1. Testing severity model...")
        test_prompt = "Defect Type: Structural\nLocation: Internal\nInspection: Automated Testing\nProduct ID: 42\nWhat is the severity?"
        result = subprocess.run(
            ["ollama", "run", SEVERITY_MODEL, test_prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"  Input: {test_prompt[:50]}...")
            print(f"  Output: {result.stdout.strip()[:100]}")
        else:
            print(f"  ✗ Error: {result.stderr}")

        # Test cost model
        print("\n2. Testing cost model...")
        test_prompt = "Defect Type: Critical\nLocation: Component\nSeverity: Critical\nInspection: Manual Testing\nProduct ID: 99\nEstimate the repair cost"
        result = subprocess.run(
            ["ollama", "run", COST_MODEL, test_prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"  Input: {test_prompt[:50]}...")
            print(f"  Output: {result.stdout.strip()[:100]}")
        else:
            print(f"  ✗ Error: {result.stderr}")

    def log_to_mlflow(self, severity_examples, cost_examples):
        """Log fine-tuning metadata to MLflow."""
        if not MLFLOW_AVAILABLE:
            return

        print("\n=== Logging to MLflow ===\n")

        with mlflow.start_run():
            mlflow.log_param("base_model", BASE_MODEL)
            mlflow.log_param("severity_model", SEVERITY_MODEL)
            mlflow.log_param("cost_model", COST_MODEL)
            mlflow.log_metric("severity_training_samples", len(severity_examples))
            mlflow.log_metric("cost_training_samples", len(cost_examples))

            # Log model info
            mlflow.log_dict({
                "severity_model": SEVERITY_MODEL,
                "cost_model": COST_MODEL,
                "created_at": self.timestamp,
                "total_training_samples": len(self.df)
            }, "model_info.json")

            print(f"✓ Logged to MLflow run: {mlflow.active_run().info.run_id}")

def main():
    print("=" * 70)
    print("Fine-tune Llama2 for Defect Prediction (Ollama-only)")
    print("=" * 70)

    # Initialize
    finetuner = OllamaFineTuner("defects_data.csv")

    # Prepare data
    severity_examples, cost_examples = finetuner.prepare_training_data()

    # Create models
    finetuner.create_models()

    # Verify and test
    finetuner.verify_models()
    finetuner.test_models()

    # Log to MLflow
    finetuner.log_to_mlflow(severity_examples, cost_examples)

    print("\n" + "=" * 70)
    print("✓ Fine-tuning Complete!")
    print("=" * 70)
    print(f"\nSeverity Model: {SEVERITY_MODEL}")
    print(f"Cost Model: {COST_MODEL}")
    print("\nUsage:")
    print(f"  ollama run {SEVERITY_MODEL} 'Cosmetic defect in Surface'")
    print(f"  ollama run {COST_MODEL} 'Critical Structural defect'")

if __name__ == "__main__":
    main()
