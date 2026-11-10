# Churn MLOps — Customer Churn Predictor with Drift Monitoring

A small, deliberately simple ML model (churn classification) wrapped in a full
MLOps lifecycle: training, experiment tracking, model registry, serving,
drift monitoring, and automated retraining.

## Why this project exists

The model itself (logistic regression / XGBoost on tabular data) is not the
point. The point is everything *around* it:

```
Data → Train → Track (MLflow) → Register → Serve (FastAPI) → Monitor (Evidently) → Retrain trigger
```

## Project status

See `PROGRESS.md` for the current step and what's done so far.

## Repo structure

```
churn-mlops/
├── data/
│   ├── train.csv              # training window (not committed if large — see .gitignore)
│   └── future_simulated.csv   # simulated "live traffic" with drift
├── src/
│   ├── train.py                # trains a model, logs to MLflow
│   ├── promote_model.py         # promotes a model to "production" if it beats the current one
│   ├── api.py                   # FastAPI app serving the production model
│   └── drift_report.py          # compares live traffic vs training distribution (Evidently)
├── tests/
│   └── test_model_quality.py    # sanity check: model must beat a dumb baseline
├── Dockerfile
├── requirements.txt
├── config.yaml                  # hyperparameters & paths, read by train.py
└── .github/workflows/ci.yml     # lint + test on every push
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Download it and place it as `data/train.csv` (instructions in `data/README.md`).

## Running MLflow locally

```bash
mlflow ui
# then open http://localhost:5000
```

## Build order

1. Baseline model (no MLOps) — get something that works.
2. Reproducible training script (`src/train.py` + `config.yaml`).
3. Experiment tracking (MLflow).
4. Model registry + promotion logic.
5. Serving (`src/api.py` + Docker).
6. Simulated drift + Evidently reports.
7. Automated retrain-on-drift loop.
8. (Stretch) Full CI/CD via GitHub Actions.
