"""
Step 2 & 3: Reproducible training + MLflow experiment tracking.

Usage:
    python src/train.py --config config.yaml

TODO (build this incrementally — don't fill in everything at once):
  1. Load config.yaml (data paths, model type, hyperparameters).
  2. Load and preprocess data/train.csv.
  3. Split train/test.
  4. Train the model specified in config (start with logistic_regression).
  5. Evaluate (accuracy, F1, AUC) on the test split.
  6. Log params + metrics + model artifact to MLflow (mlflow.start_run()).
  7. Print a summary so you can see the run without opening the MLflow UI.
"""

import argparse
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    config = load_config(config_path)
    print(f"Loaded config: {config}")

    # TODO: load data (config['data']['train_path'])
    # TODO: preprocess (encode categoricals, handle missing values)
    # TODO: train/test split (config['data']['test_size'], random_state)
    # TODO: train model (config['model']['type'], config['model']['params'])
    # TODO: evaluate (accuracy, F1, AUC)
    # TODO: wrap all of the above in an mlflow.start_run() block and log:
    #   - mlflow.log_params(...)
    #   - mlflow.log_metrics(...)
    #   - mlflow.sklearn.log_model(...)  (or xgboost equivalent)

    raise NotImplementedError("Fill in the training steps above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
