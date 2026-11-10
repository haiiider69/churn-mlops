"""
first of all 
install env : python -m venv venv
activate it : venv\Scripts\activate
then : pip install -r requirements.txt


Step 2 & 3: Reproducible training + MLflow experiment tracking.

Usage:(initilazing training)
    python src/train.py --config config.yaml
"""

import argparse
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_and_preprocess(data_path: str, target_column: str):
    df = pd.read_csv(data_path)

    # Quirk 1: TotalCharges is text with ~11 blank-string rows (tenure=0 customers).
    # Coerce to numeric; blanks become NaN, then fill with 0 (they haven't been charged yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Quirk 3: customerID is just an identifier, not a feature.
    df = df.drop(columns=["customerID"])

    # Quirk 2: target column is "Yes"/"No" -> 1/0
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

    # Encode remaining categorical columns (simple label encoding for the baseline —
    # one-hot would be better long-term, but label encoding is fine to get a first
    # working model quickly).
    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def main(config_path: str):
    config = load_config(config_path)
    print(f"Loaded config: {config}")

    data_cfg = config["data"]
    model_cfg = config["model"]
    mlflow_cfg = config["mlflow"]

    X, y = load_and_preprocess(data_cfg["train_path"], data_cfg["target_column"])
    print(f"Data shape after preprocessing: {X.shape}")
    print(f"Class balance:\n{y.value_counts(normalize=True)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,  # keep churn/no-churn ratio consistent in both splits
    )

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run():
        params = model_cfg["params"]
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": auc})
        mlflow.sklearn.log_model(model, "model")

        print(f"\n--- Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"ROC AUC:  {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)