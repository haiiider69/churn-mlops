"""
Step 4: Model registry + promotion logic.

Goal: only mark a new model as "production" if it beats the current
production model on the metric you care about (e.g., F1 score).

TODO:
  1. Connect to the MLflow tracking server / local mlruns folder.
  2. Find the latest run's metric (e.g., F1) for the newly trained model.
  3. Find the current "production"-tagged model's metric.
  4. If new > current (or no production model exists yet): promote it
     (set the MLflow registry stage to "Production", or use an alias).
  5. Otherwise: log that the new model did not beat production, no change made.

Usage (once implemented):
    python src/promote_model.py --metric f1_score
"""

import argparse


def main(metric_name: str):
    # TODO: implement promotion logic described above
    raise NotImplementedError("Fill in promotion logic.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="f1_score")
    args = parser.parse_args()
    main(args.metric)
