"""Enregistrement des évaluations dans MLflow."""

import json
import math
from pathlib import Path

import mlflow
import pandas as pd

from assistant_rag.config import MLFLOW_ARTIFACTS, MLFLOW_DB, MLFLOW_EXPERIMENT
from assistant_rag.evaluation.judge import VERDICT_SCORES
from assistant_rag.evaluation.metrics import score_records, summary_by_type
from assistant_rag.evaluation.runner import ANSWERS_FILE, load_records

CONFIG_FILE = "config.json"
REVIEW_FILE = "revue_humaine.csv"


def tracking_uri(db_path: Path = MLFLOW_DB) -> str:
    return f"sqlite:///{db_path.as_posix()}"


def setup_experiment(db_path: Path = MLFLOW_DB, artifacts_dir: Path = MLFLOW_ARTIFACTS) -> str:
    """Configure MLflow sur une base SQLite locale et renvoie l'identifiant de l'expérience."""
    mlflow.set_tracking_uri(tracking_uri(db_path))
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is not None:
        return experiment.experiment_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=artifacts_dir.resolve().as_uri())


def run_params(config: dict) -> dict[str, str]:
    params = {key: value for key, value in config.items() if key not in ("settings", "started_at")}
    params.update({f"settings.{key}": value for key, value in config.get("settings", {}).items()})
    return {key: str(value) for key, value in params.items()}


def run_metrics(records: list[dict]) -> dict[str, float]:
    """Métriques automatiques au format MLflow : « <type>.<métrique> »."""
    table = summary_by_type(records)
    return {
        f"{group}.{metric}": float(value)
        for group, row in table.iterrows()
        for metric, value in row.items()
        if not (isinstance(value, float) and math.isnan(value))
    }


def human_metrics(records: list[dict], review: pd.DataFrame) -> dict[str, float]:
    """Exactitude recalculée avec les verdicts de la relecture humaine et accord du juge avec celle-ci."""
    scores = score_records(records).set_index("id")
    answerable = scores[scores["type"] != "hors_corpus"]
    human = answerable["correctness"].copy()
    human.loc[review.index] = review["verdict_humain"].map(VERDICT_SCORES)
    metrics = {
        f"{kind}.exactitude_relue": float(value) for kind, value in human.groupby(answerable["type"]).mean().items()
    }
    metrics["ensemble.exactitude_relue"] = float(human.mean())
    metrics["accord_juge_relecture"] = float((review["verdict_juge"] == review["verdict_humain"]).mean())
    return metrics


def log_results_dir(run_dir: Path, db_path: Path = MLFLOW_DB, artifacts_dir: Path = MLFLOW_ARTIFACTS) -> str:
    """Enregistre une évaluation de results/ comme run MLflow, en remplaçant un run déjà enregistré pour ce dossier."""
    experiment_id = setup_experiment(db_path, artifacts_dir)
    for run_id in mlflow.search_runs([experiment_id], filter_string=f"tags.results_dir = '{run_dir.name}'")["run_id"]:
        mlflow.delete_run(run_id)

    config = json.loads((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    records = load_records(run_dir / ANSWERS_FILE)
    metrics = run_metrics(records)
    review_path = run_dir / REVIEW_FILE
    if review_path.exists():
        metrics.update(human_metrics(records, pd.read_csv(review_path).set_index("id")))

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_dir.name) as run:
        mlflow.set_tags({"results_dir": run_dir.name, "system": config["system"], "split": config["split"]})
        mlflow.log_params(run_params(config))
        mlflow.log_metrics(metrics)
        for path in sorted(run_dir.iterdir()):
            mlflow.log_artifact(str(path))
    return run.info.run_id
