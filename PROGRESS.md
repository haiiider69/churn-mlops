# Progress Tracker

Update this file as you complete each step — it's the single source of truth
for "where am I in this project."

- [ ] Step 1 — Baseline model (notebook or quick script, no MLOps)
- [ ] Step 2 — Reproducible training script (`src/train.py` + `config.yaml`)
- [ ] Step 3 — Experiment tracking with MLflow
- [ ] Step 4 — Model registry + promotion logic (`src/promote_model.py`)
- [ ] Step 5 — Serving via FastAPI + Docker
- [ ] Step 6 — Simulated drift data + Evidently drift report
- [ ] Step 7 — Automated retrain-on-drift trigger
- [ ] Step 8 — (Stretch) Full CI/CD via GitHub Actions

## Notes / decisions log

- Dataset: Telco Customer Churn (Kaggle, ~7k rows, binary classification)
- Model choice: TBD (start with logistic regression as baseline, try XGBoost after)
