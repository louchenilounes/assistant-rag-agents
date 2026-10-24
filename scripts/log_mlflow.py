"""Enregistre toutes les évaluations du dossier results/ dans MLflow."""

import mlflow
import pandas as pd

from assistant_rag.config import RESULTS_DIR
from assistant_rag.tracking import CONFIG_FILE, log_results_dir, setup_experiment, tracking_uri

COLUMNS = {
    "tags.results_dir": "évaluation",
    "metrics.ensemble.retrieval_recall": "rappel@k",
    "metrics.ensemble.citation_precision": "précision citations",
    "metrics.ensemble.correctness": "exactitude (juge)",
    "metrics.ensemble.exactitude_relue": "exactitude relue",
    "metrics.ensemble.llm_calls": "appels LLM",
    "metrics.ensemble.latency": "latence (s)",
}


def main() -> None:
    run_dirs = sorted(path for path in RESULTS_DIR.iterdir() if (path / CONFIG_FILE).exists())
    for run_dir in run_dirs:
        log_results_dir(run_dir)
        print(f"Enregistré : {run_dir.name}")

    runs = mlflow.search_runs([setup_experiment()])
    table = runs.reindex(columns=list(COLUMNS)).rename(columns=COLUMNS).sort_values("évaluation")
    with pd.option_context("display.width", 200, "display.max_columns", 10, "display.precision", 2):
        print(table.to_string(index=False))
    print(f"\nInterface : mlflow ui --backend-store-uri {tracking_uri()}")


if __name__ == "__main__":
    main()
