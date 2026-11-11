"""
Step 4: Model registry + promotion logic.

Finds the best run (by F1 score) in the MLflow experiment, and promotes it
to the "production" alias only if it beats the currently-promoted model.

Usage:
    python src/promote_model.py --config config.yaml
"""

import argparse
import yaml
import mlflow
from mlflow import MlflowClient


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_best_run(client: MlflowClient, experiment_name: str, metric: str):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No runs found in this experiment.")
    return runs[0]


def get_current_production_metric(client: MlflowClient, model_name: str, metric: str):
    """Returns the metric value tagged on the current 'production' alias, or None
    if no model is registered / no production alias exists yet."""
    try:
        version = client.get_model_version_by_alias(model_name, "production")
    except mlflow.exceptions.MlflowException:
        return None, None

    metric_value = version.tags.get(metric)
    return (float(metric_value) if metric_value is not None else None), version


def main(config_path: str):
    config = load_config(config_path)
    mlflow_cfg = config["mlflow"]
    model_name = config["registry"]["model_name"]
    metric = "f1_score"

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = MlflowClient()

    best_run = get_best_run(client, mlflow_cfg["experiment_name"], metric)
    best_metric_value = best_run.data.metrics[metric]
    print(f"Best run found: {best_run.info.run_id} ({metric}={best_metric_value:.4f})")

    current_metric_value, current_version = get_current_production_metric(
        client, model_name, metric
    )

    if current_metric_value is not None:
        print(f"Current production model: version {current_version.version} "
              f"({metric}={current_metric_value:.4f})")

    if current_metric_value is not None and best_metric_value <= current_metric_value:
        print(f"\nNo promotion: best run ({best_metric_value:.4f}) does not beat "
              f"current production ({current_metric_value:.4f}).")
        return

    # Register the best run's model as a new version.
    model_uri = f"runs:/{best_run.info.run_id}/model"
    new_version = mlflow.register_model(model_uri, model_name)

    # Tag the new version with its metric so future comparisons don't need
    # to re-query the run.
    client.set_model_version_tag(model_name, new_version.version, metric, str(best_metric_value))

    # Point the "production" alias at this new version.
    client.set_registered_model_alias(model_name, "production", new_version.version)

    print(f"\nPromoted run {best_run.info.run_id} to production "
          f"as version {new_version.version} ({metric}={best_metric_value:.4f}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)