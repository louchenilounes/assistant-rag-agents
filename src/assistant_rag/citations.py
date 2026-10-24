"""Lecture des citations [n] dans les réponses générées."""

import re

REFUSAL = "Je ne trouve pas la réponse dans les documents fournis."

CITATION_GROUP = re.compile(r"\[(\s*\d+\s*(?:[,;]\s*\d+\s*)*)\]")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Retire un éventuel raisonnement <think>...</think> laissé par le modèle."""
    text = THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def parse_citations(text: str, source_count: int) -> list[int]:
    """Renvoie les numéros de sources cités, sans doublon, dans l'ordre croissant.

    Les numéros hors de l'intervalle [1, source_count] sont ignorés.
    """
    numbers = {
        int(value) for group in CITATION_GROUP.findall(text) for value in re.split(r"[,;]", group) if value.strip()
    }
    return sorted(number for number in numbers if 1 <= number <= source_count)


def is_refusal(text: str) -> bool:
    """Indique si la réponse est un refus de répondre faute d'information dans les sources."""
    normalized = " ".join(text.lower().split())
    return REFUSAL.lower().rstrip(".") in normalized


def is_refused_answer(text: str, source_count: int) -> bool:
    """Réponse entièrement refusée : phrase de refus sans aucune citation.

    Une réponse qui traite une partie de la question avec des citations puis signale
    qu'un aspect n'est pas couvert reste une réponse, évaluée comme telle.
    """
    return is_refusal(text) and not parse_citations(text, source_count)
