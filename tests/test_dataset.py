from collections import Counter

from assistant_rag.config import DOCUMENTS
from assistant_rag.evaluation.dataset import QUESTION_TYPES, SPLITS, load_questions


def test_questions_file_is_consistent():
    questions = load_questions()

    assert len(questions) == 60
    assert len({question.id for question in questions}) == len(questions)
    assert {question.type for question in questions} <= set(QUESTION_TYPES)
    assert {question.split for question in questions} == set(SPLITS)


def test_dev_and_test_splits_cover_every_question_type():
    counts = Counter((question.split, question.type) for question in load_questions())

    assert all(counts[(split, kind)] > 0 for split in SPLITS for kind in QUESTION_TYPES)
    assert sum(count for (split, _), count in counts.items() if split == "dev") == 40


def test_expected_articles_exist_and_match_question_type():
    for question in load_questions():
        if question.type == "hors_corpus":
            assert question.expected == () and question.reference == ""
            continue
        assert question.reference
        assert len(question.expected) == (1 if question.type == "factuelle" else 2)
        for document, article in question.expected:
            assert 1 <= article <= DOCUMENTS[document].expected_articles


def test_load_questions_filters_by_split():
    assert {question.split for question in load_questions(split="test")} == {"test"}
