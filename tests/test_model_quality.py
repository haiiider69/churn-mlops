"""
Sanity check used by CI (Step 8): the trained model must beat a dumb
baseline (e.g., always predict the majority class). This is NOT a
replacement for proper evaluation — it's a cheap guard against a
completely broken training pipeline.

TODO:
  1. Load the test split (same split used in train.py — consider saving
     it to disk during training so this test uses the exact same data).
  2. Compute the majority-class baseline accuracy.
  3. Load the latest trained model's accuracy (from MLflow or a saved
     metrics file).
  4. Assert the model's accuracy is meaningfully above baseline.
"""


def test_model_beats_baseline():
    # TODO: implement as described above
    raise NotImplementedError("Fill in once train.py produces a model + metrics.")
