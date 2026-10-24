"""Évaluation de l'exactitude des réponses par un LLM-juge."""

import json
from dataclasses import dataclass

from llama_index.core.llms import LLM, ChatMessage
from llama_index.llms.ollama import Ollama

from assistant_rag.citations import strip_thinking
from assistant_rag.config import Settings

VERDICT_SCORES = {"correct": 1.0, "partiel": 0.5, "incorrect": 0.0}
JUDGE_PROMPT_VERSION = "juge-v2"

JUDGE_PROMPT = """Tu évalues la réponse d'un assistant à une question sur le RGPD ou sur l'AI Act.
Compare la réponse évaluée à la réponse de référence, qui est exacte.

Verdicts possibles :
- "correct" : la réponse contient les éléments essentiels de la référence et ne contient pas d'erreur.
- "partiel" : la réponse est en partie exacte mais omet un élément essentiel ou contient une imprécision.
- "incorrect" : la réponse est fausse, hors sujet, ou affirme ne pas connaître la réponse.

Juge le sens et non la formulation : « oui, à condition que… » peut équivaloir à « en principe non, sauf… ».
Des informations supplémentaires exactes ne sont pas une erreur et ne justifient pas un verdict « partiel ».
Ignore le style, la longueur et les numéros de citation entre crochets.
Réponds uniquement avec un objet JSON de la forme {"verdict": "...", "justification": "..."}.
La justification tient en une phrase."""


@dataclass(frozen=True)
class Judgment:
    verdict: str
    score: float | None
    justification: str


def build_judge_llm(settings: Settings) -> LLM:
    return Ollama(
        model=settings.judge_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        context_window=settings.context_window,
        request_timeout=settings.request_timeout,
        json_mode=True,
        thinking=False,
    )


def parse_judgment(text: str) -> Judgment:
    """Lit la réponse JSON du juge ; un verdict illisible n'est pas compté dans l'exactitude."""
    try:
        data = json.loads(strip_thinking(text))
        verdict = str(data["verdict"]).strip().lower()
        justification = str(data.get("justification", "")).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return Judgment("invalide", None, text.strip()[:300])
    if verdict not in VERDICT_SCORES:
        return Judgment("invalide", None, justification)
    return Judgment(verdict, VERDICT_SCORES[verdict], justification)


class Judge:
    def __init__(self, llm: LLM):
        self.llm = llm

    def judge(self, question: str, reference: str, answer: str) -> Judgment:
        content = f"Question : {question}\n\nRéponse de référence : {reference}\n\nRéponse évaluée : {answer}"
        messages = [ChatMessage(role="system", content=JUDGE_PROMPT), ChatMessage(role="user", content=content)]
        return parse_judgment(self.llm.chat(messages).message.content or "")
