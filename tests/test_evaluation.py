import json

import pytest

from assistant_rag.evaluation.dataset import Question
from assistant_rag.evaluation.judge import parse_judgment
from assistant_rag.evaluation.metrics import score_record, summarize
from assistant_rag.evaluation.runner import load_records, run_evaluation
from assistant_rag.rag import Answer, Source


def make_record(**overrides) -> dict:
    record = {
        "id": "q1",
        "type": "multi_articles",
        "expected": [["rgpd", 33], ["rgpd", 34]],
        "retrieved": [
            {"number": 1, "document": "rgpd", "article": 12},
            {"number": 2, "document": "rgpd", "article": 33},
            {"number": 3, "document": "rgpd", "article": 33},
        ],
        "cited": [{"number": 2, "document": "rgpd", "article": 33}, {"number": 1, "document": "rgpd", "article": 12}],
        "refused": False,
        "latency": 10.0,
        "judge": {"verdict": "partiel", "score": 0.5, "justification": ""},
    }
    record.update(overrides)
    return record


def test_score_record_computes_retrieval_and_citation_metrics():
    scores = score_record(make_record())

    assert scores["hit"] == 1.0
    assert scores["reciprocal_rank"] == 0.5
    assert scores["retrieval_recall"] == 0.5
    assert scores["citation_precision"] == 0.5
    assert scores["citation_recall"] == 0.5
    assert scores["correctness"] == 0.5
    assert scores["false_refusal"] == 0.0


def test_refusal_on_answerable_question_counts_as_incorrect():
    scores = score_record(make_record(refused=True, cited=[], judge=None))

    assert scores["correctness"] == 0.0
    assert scores["false_refusal"] == 1.0
    assert scores["citation_precision"] is None
    assert scores["citation_recall"] == 0.0


def test_out_of_corpus_question_only_measures_refusal():
    scores = score_record(make_record(type="hors_corpus", expected=[], refused=True, judge=None))

    assert scores["correct_refusal"] == 1.0
    assert scores["hit"] is None and scores["correctness"] is None


def test_summarize_reports_each_question_type_and_overall():
    records = [make_record(), make_record(id="q2", type="hors_corpus", expected=[], refused=False, judge=None)]

    table = summarize(records)

    assert list(table.index) == ["Multi-articles", "Hors corpus", "Ensemble"]
    assert table.loc["Ensemble", "Questions"] == 2
    assert table.loc["Hors corpus", "Refus correct"] == 0.0
    assert table.loc["Ensemble", "Exactitude"] == 0.5


@pytest.mark.parametrize(
    ("text", "verdict", "score"),
    [
        ('{"verdict": "correct", "justification": "Complet."}', "correct", 1.0),
        ('{"verdict": "Partiel", "justification": "Incomplet."}', "partiel", 0.5),
        ('{"verdict": "faux"}', "invalide", None),
        ("pas du JSON", "invalide", None),
    ],
)
def test_parse_judgment(text, verdict, score):
    judgment = parse_judgment(text)

    assert (judgment.verdict, judgment.score) == (verdict, score)


class FixedSystem:
    def __init__(self):
        self.calls = 0

    def answer(self, question: str) -> Answer:
        self.calls += 1
        source = Source(1, "rgpd-art033-00", "rgpd", "RGPD", 33, "Notification", 52, 52, "", 0.9)
        return Answer(question=question, text="Réponse [1].", sources=[source], cited=[1], latency=1.0)


QUESTIONS = [
    Question("q1", "dev", "factuelle", "Délai de notification ?", "72 heures.", (("rgpd", 33),)),
    Question("q2", "dev", "hors_corpus", "Durée du travail ?", "", ()),
]


def test_run_evaluation_saves_each_answer_and_resumes(tmp_path):
    system = FixedSystem()

    run_evaluation(system, QUESTIONS[:1], tmp_path)
    records = run_evaluation(system, QUESTIONS, tmp_path)

    assert system.calls == 2
    assert [record["id"] for record in records] == ["q1", "q2"]
    assert load_records(tmp_path / "answers.jsonl") == records
    assert records[0]["cited"] == [{"number": 1, "document": "rgpd", "article": 33}]
    assert (tmp_path / "metrics.csv").exists()
    assert json.loads((tmp_path / "answers.jsonl").read_text(encoding="utf-8").splitlines()[1])["id"] == "q2"
