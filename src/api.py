"""
Step 5: Serve the production model via FastAPI.

TODO:
  1. Load the current "production" model from the MLflow registry on startup.
  2. Define a Pydantic model describing the input features (match config.yaml's
     expected columns, minus the target).
  3. Implement POST /predict: takes the input features, returns churn
     probability + predicted class.
  4. (Step 6 hook) Log every incoming request's features to a file or DB —
     this becomes the "live traffic" data that drift_report.py compares
     against the training distribution.

Usage (once implemented):
    uvicorn src.api:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Predictor API")


# TODO: replace with the real feature set from your dataset
class CustomerFeatures(BaseModel):
    tenure: int
    monthly_charges: float
    contract_type: str
    # ... add remaining features from the Telco dataset


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: CustomerFeatures):
    # TODO: load production model (once, at startup — not per request)
    # TODO: transform `features` into the model's expected input format
    # TODO: run prediction, return probability + class
    # TODO: append `features` to a log file for drift monitoring (Step 6)
    raise NotImplementedError("Fill in prediction logic.")
