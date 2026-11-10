"""
Step 6: Drift monitoring with Evidently.

Goal: compare the distribution of "live traffic" (simulated future data, or
logged real requests from the API) against the distribution the model was
trained on, and flag when they've diverged meaningfully.

TODO:
  1. Load the original training data (the reference distribution).
  2. Load the simulated "future" data or logged live-traffic data (the
     current distribution).
  3. Use Evidently's DataDriftPreset (or similar) to generate a report.
  4. Save the report as HTML so you can open and inspect it visually.
  5. Extract a simple pass/fail signal (e.g., "drift detected: true/false")
     that Step 7's retrain trigger can read programmatically.

Usage (once implemented):
    python src/drift_report.py --reference data/train.csv --current data/future_simulated.csv
"""

import argparse


def main(reference_path: str, current_path: str):
    # TODO: implement the Evidently comparison described above
    raise NotImplementedError("Fill in drift detection logic.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default="data/train.csv")
    parser.add_argument("--current", type=str, default="data/future_simulated.csv")
    args = parser.parse_args()
    main(args.reference, args.current)
