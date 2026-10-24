"""Chargement du jeu de questions annoté."""

import json
from dataclasses import dataclass
from pathlib import Path

from assistant_rag.config import QUESTIONS_PATH

QUESTION_TYPES = ("factuelle", "multi_articles", "hors_corpus")
SPLITS = ("dev", "test")


@dataclass(frozen=True)
class Question:
    """Question annotée : réponse de référence et articles qui la justifient."""

    id: str
    split: str
    type: str
    question: str
    reference: str
    expected: tuple[tuple[str, int], ...]

    @property
    def answerable(self) -> bool:
        return self.type != "hors_corpus"

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        return cls(
            id=data["id"],
            split=data["split"],
            type=data["type"],
            question=data["question"],
            reference=data["reference"],
            expected=tuple((item["document"], int(item["article"])) for item in data["expected"]),
        )


def load_questions(path: Path = QUESTIONS_PATH, split: str | None = None) -> list[Question]:
    """Charge les questions, éventuellement limitées à un sous-ensemble (dev ou test)."""
    with path.open(encoding="utf-8") as file:
        questions = [Question.from_dict(json.loads(line)) for line in file if line.strip()]
    if split is not None:
        questions = [question for question in questions if question.split == split]
    return questions
