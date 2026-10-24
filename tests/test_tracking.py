import json

import mlflow
import pandas as pd

from assistant_rag.tracking import human_metrics, log_results_dir, run_metrics, run_params, setup_experiment

RECORDS = [
    {
        "id": "q1",
        "type": "factuelle",
        "expected": [["rgpd", 33]],
        "retrieved": [{"number": 1, "document": "rgpd", "article": 33}],
        "cited": [{"number": 1, "document": "rgpd", "article": 33}],
        "refused": False,
        "latency": 4.0,
        "llm_calls": 1,
        "judge": {"verdict": "partiel", "score": 0.5, "justification": ""},
    },
    {
        "id": "q2",
        "type": "hors_corpus",
        "expected": [],
        "retrieved": [{"number": 1, "document": "ai_act", "article": 5}],
        "cited": [],
        "refused": True,
        "latency": 2.0,
        "llm_calls": 1,
        "judge": None,
    },
]


def make_run_dir(tmp_path, name="simple-dev"):
    run_dir = tmp_path / "results" / name
    run_dir.mkdir(parents=True)
    config = {"system": "simple", "split": "dev", "settings": {"top_k": 5}, "started_at": "2026-01-01T00:00:00"}
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in RECORDS), encoding="utf-8")
    pd.DataFrame([{"id": "q1", "verdict_juge": "partiel", "verdict_humain": "correct", "commentaire": ""}]).to_csv(
        run_dir / "revue_humaine.csv", index=False
    )
    return run_dir


def test_run_params_flattens_settings_and_drops_timestamp():
    params = run_params({"system": "agents", "settings": {"top_k": 5}, "started_at": "x", "judge": None})

    assert params == {"system": "agents", "settings.top_k": "5", "judge": "None"}


def test_run_metrics_are_named_by_question_type():
    metrics = run_metrics(RECORDS)

    assert metrics["factuelle.correctness"] == 0.5
    assert metrics["hors_corpus.correct_refusal"] == 1.0
    assert metrics["ensemble.questions"] == 2
    assert "hors_corpus.correctness" not in metrics


def test_human_metrics_use_reviewed_verdicts():
    review = pd.DataFrame({"verdict_juge": ["partiel"], "verdict_humain": ["correct"]}, index=["q1"])

    metrics = human_metrics(RECORDS, review)

    assert metrics == {
        "factuelle.exactitude_relue": 1.0,
        "ensemble.exactitude_relue": 1.0,
        "accord_juge_relecture": 0.0,
    }


def test_log_results_dir_replaces_previous_run(tmp_path):
    run_dir = make_run_dir(tmp_path)
    db_path, artifacts_dir = tmp_path / "mlflow.db", tmp_path / "artifacts"

    log_results_dir(run_dir, db_path, artifacts_dir)
    run_id = log_results_dir(run_dir, db_path, artifacts_dir)

    runs = mlflow.search_runs([setup_experiment(db_path, artifacts_dir)])
    assert list(runs["run_id"]) == [run_id]
    assert runs.loc[0, "metrics.ensemble.exactitude_relue"] == 1.0
    assert runs.loc[0, "params.settings.top_k"] == "5"
    artifacts = {item.path for item in mlflow.MlflowClient().list_artifacts(run_id)}
    assert artifacts == {"answers.jsonl", "config.json", "revue_humaine.csv"}
