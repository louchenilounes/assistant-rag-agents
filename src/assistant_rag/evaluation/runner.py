"""Exécution d'une évaluation, reprise possible après interruption."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from assistant_rag.evaluation.dataset import Question
from assistant_rag.evaluation.judge import Judge
from assistant_rag.evaluation.metrics import summarize
from assistant_rag.rag import Answer

ANSWERS_FILE = "answers.jsonl"
METRICS_FILE = "metrics.csv"


class QASystem(Protocol):
    def answer(self, question: str) -> Answer: ...


def build_record(question: Question, answer: Answer, judge: Judge | None) -> dict:
    record = {
        "id": question.id,
        "split": question.split,
        "type": question.type,
        "question": question.question,
        "reference": question.reference,
        "expected": [list(article) for article in question.expected],
        "answer": answer.text,
        "refused": answer.refused,
        "retrieved": [
            {"number": s.number, "document": s.document, "article": s.article, "score": round(s.score, 4)}
            for s in answer.sources
        ],
        "cited": [{"number": s.number, "document": s.document, "article": s.article} for s in answer.cited_sources],
        "latency": round(answer.latency, 2),
        "judge": None,
    }
    if judge is not None and question.answerable and not answer.refused:
        record["judge"] = asdict(judge.judge(question.question, question.reference, answer.text))
    return record


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_evaluation(
    system: QASystem, questions: list[Question], output_dir: Path, judge: Judge | None = None
) -> list[dict]:
    """Évalue les questions non encore traitées et enregistre chaque résultat dès qu'il est obtenu."""
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / ANSWERS_FILE
    done = {record["id"] for record in load_records(answers_path)}
    remaining = [question for question in questions if question.id not in done]
    print(f"{len(done)} questions déjà évaluées, {len(remaining)} restantes")

    for position, question in enumerate(remaining, start=1):
        record = build_record(question, system.answer(question.question), judge)
        with answers_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        verdict = record["judge"]["verdict"] if record["judge"] else ("refus" if record["refused"] else "-")
        print(f"[{position}/{len(remaining)}] {question.id} — {record['latency']:.1f} s — {verdict}")

    ids = {question.id for question in questions}
    records = [record for record in load_records(answers_path) if record["id"] in ids]
    summarize(records).to_csv(output_dir / METRICS_FILE)
    return records
