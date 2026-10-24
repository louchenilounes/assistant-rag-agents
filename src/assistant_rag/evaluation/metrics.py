"""Calcul des métriques par question et agrégées."""

from statistics import mean

import pandas as pd

GROUP_LABELS = {
    "factuelle": "Factuelles",
    "multi_articles": "Multi-articles",
    "hors_corpus": "Hors corpus",
}

METRIC_LABELS = {
    "questions": "Questions",
    "hit": "Hit@k",
    "retrieval_recall": "Rappel@k",
    "reciprocal_rank": "MRR",
    "citation_precision": "Précision des citations",
    "citation_recall": "Rappel des citations",
    "correctness": "Exactitude",
    "false_refusal": "Refus à tort",
    "correct_refusal": "Refus correct",
    "latency": "Latence moyenne (s)",
    "llm_calls": "Appels au LLM",
}


def _articles(items: list[dict]) -> list[tuple[str, int]]:
    return [(item["document"], int(item["article"])) for item in items]


def score_record(record: dict) -> dict:
    """Métriques d'une question évaluée. Une métrique sans objet vaut None."""
    expected = {(document, int(article)) for document, article in record["expected"]}
    retrieved = _articles(record["retrieved"])
    cited = _articles(record["cited"])
    scores = {
        "id": record["id"],
        "type": record["type"],
        "latency": record["latency"],
        "llm_calls": record.get("llm_calls", 1),
        "hit": None,
        "retrieval_recall": None,
        "reciprocal_rank": None,
        "citation_precision": None,
        "citation_recall": None,
        "correctness": None,
        "false_refusal": None,
        "correct_refusal": None,
    }
    if not expected:
        scores["correct_refusal"] = float(record["refused"])
        return scores

    ranks = [rank for rank, article in enumerate(retrieved, start=1) if article in expected]
    scores["hit"] = float(bool(ranks))
    scores["reciprocal_rank"] = 1 / ranks[0] if ranks else 0.0
    scores["retrieval_recall"] = len(expected & set(retrieved)) / len(expected)
    scores["citation_recall"] = len(expected & set(cited)) / len(expected)
    if cited:
        scores["citation_precision"] = mean(article in expected for article in cited)
    scores["false_refusal"] = float(record["refused"])
    judgment = record.get("judge")
    if record["refused"]:
        scores["correctness"] = 0.0
    elif judgment is not None:
        scores["correctness"] = judgment["score"]
    return scores


def score_records(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([score_record(record) for record in records])


def summary_by_type(records: list[dict]) -> pd.DataFrame:
    """Métriques moyennes par type de question et sur l'ensemble, avec les noms internes des métriques."""
    scores = score_records(records)
    metrics = [name for name in METRIC_LABELS if name != "questions"]
    groups = {key: scores[scores["type"] == key] for key in GROUP_LABELS}
    groups["ensemble"] = scores
    rows = {}
    for key, frame in groups.items():
        if frame.empty:
            continue
        row = {"questions": len(frame)}
        row.update({name: pd.to_numeric(frame[name], errors="coerce").mean() for name in metrics})
        rows[key] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def summarize(records: list[dict]) -> pd.DataFrame:
    """Tableau des métriques par type de question et sur l'ensemble des questions, avec des libellés lisibles."""
    labels = {**GROUP_LABELS, "ensemble": "Ensemble"}
    return summary_by_type(records).rename(index=labels, columns=METRIC_LABELS)
