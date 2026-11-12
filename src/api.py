"""
Step 5: Serve the production model via FastAPI.

Usage:
    uvicorn src.api:app --reload
"""

import os
import joblib
import yaml
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Churn Predictor API")

# --- Load config ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
MODEL_NAME = config["registry"]["model_name"]

# --- Load production model + preprocessing artifacts at startup ---
from mlflow import MlflowClient

client = MlflowClient()
version = client.get_model_version_by_alias(MODEL_NAME, "production")
model_type = version.tags.get("model_type", "xgboost")

model_uri = f"models:/{MODEL_NAME}@production"
if model_type == "xgboost":
    import mlflow.xgboost
    model = mlflow.xgboost.load_model(model_uri)
else:
    import mlflow.sklearn
    model = mlflow.sklearn.load_model(model_uri)

artifacts_dir = client.download_artifacts(version.run_id, "preprocessing")
scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(artifacts_dir, "label_encoders.pkl"))
feature_columns = joblib.load(os.path.join(artifacts_dir, "feature_columns.pkl"))

print(f"Loaded production model: version {version.version} ({model_type})")


# --- Request schema: every feature the model expects, as raw (unencoded) values ---
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def preprocess(features: CustomerFeatures) -> pd.DataFrame:
    row = features.model_dump()
    df = pd.DataFrame([row])

    for col, le in label_encoders.items():
        value = df.at[0, col]
        if value not in le.classes_:
            raise HTTPException(
                status_code=400,
                detail=f"Unrecognized value '{value}' for field '{col}'. "
                       f"Expected one of: {list(le.classes_)}",
            )
        df[col] = le.transform(df[col])

    df = df[feature_columns]  # enforce exact training column order
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_columns)
    return df_scaled


@app.get("/health")
def health():
    return {"status": "ok", "model_version": version.version, "model_type": model_type}


@app.post("/predict")
def predict(features: CustomerFeatures):
    X = preprocess(features)
    proba = model.predict_proba(X)[0][1]
    prediction = int(proba >= 0.5)

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": prediction,
    }