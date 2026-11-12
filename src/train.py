"""
Step 2 & 3: Reproducible training + MLflow experiment tracking.

Usage:
    python src/train.py --config config.yaml
    activate it : venv\Scripts\activate

"""

import argparse
import os
import shutil
import joblib
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_and_preprocess(data_path: str, target_column: str):
    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop(columns=["customerID"])
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include="str").columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=[target_column])
    y = df[target_column]
    feature_columns = list(X.columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, label_encoders, scaler, feature_columns


def build_model(model_type: str, config: dict):
    if model_type == "logistic_regression":
        params = config["model"]["logistic_regression"]
        return LogisticRegression(**params), params
    elif model_type == "xgboost":
        params = config["model"]["xgboost"]
        return XGBClassifier(**params, eval_metric="logloss"), params
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def log_model(model, model_type: str):
    if model_type == "xgboost":
        mlflow.xgboost.log_model(model, "model")
    else:
        mlflow.sklearn.log_model(model, "model")


def main(config_path: str):
    config = load_config(config_path)
    print(f"Loaded config: {config}")

    data_cfg = config["data"]
    model_type = config["model"]["type"]
    mlflow_cfg = config["mlflow"]

    X, y, label_encoders, scaler, feature_columns = load_and_preprocess(
        data_cfg["train_path"], data_cfg["target_column"]
    )
    print(f"Data shape after preprocessing: {X.shape}")
    print(f"Class balance:\n{y.value_counts(normalize=True)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run():
        model, params = build_model(model_type, config)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": auc})
        log_model(model, model_type)

        # --- Persist preprocessing so serving can reproduce it exactly ---
        tmp_dir = "artifacts_tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(tmp_dir, "scaler.pkl"))
        joblib.dump(label_encoders, os.path.join(tmp_dir, "label_encoders.pkl"))
        joblib.dump(feature_columns, os.path.join(tmp_dir, "feature_columns.pkl"))
        mlflow.log_artifacts(tmp_dir, artifact_path="preprocessing")
        shutil.rmtree(tmp_dir)

        print(f"\n--- Results ({model_type}) ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"ROC AUC:  {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)