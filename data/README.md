# Data

This folder holds the training data and the simulated "future" data used
for drift monitoring. The raw CSVs are **not** committed to Git (see
`.gitignore`) to keep the repo small — download them yourself:

1. Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   from Kaggle.
2. Place it here as `train.csv`.
3. `future_simulated.csv` will be generated later (Step 6) by splitting off
   part of the data and artificially shifting its distribution to simulate
   drift — there's a script for this once you reach that step.
