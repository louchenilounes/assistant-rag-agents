"""Agents chercheur, rédacteur et vérificateur, reliés par un graphe LangGraph."""

import json
import re
import time
from collections import Counter
from dataclasses import replace

from langgraph.graph import END, START, StateGraph
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM

from assistant_rag.agents.prompts import (
    COVERAGE_INSTRUCTIONS,
    RESEARCHER_PROMPT,
    REVISION_INSTRUCTIONS,
    VERIFIER_PROMPT,
)
from assistant_rag.agents.state import AgentState
from assistant_rag.citations import is_refused_answer, parse_citations
from assistant_rag.config import Settings
from assistant_rag.indexing import get_index
from assistant_rag.prompts import SYSTEM_PROMPT
from assistant_rag.rag import Answer, Source, build_llm, build_user_prompt, chat, format_context

VERDICTS = ("valide", "a_corriger", "incomplete")
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict:
    """Lit un objet JSON produit par un LLM ; renvoie un dictionnaire vide s'il est illisible."""
    for candidate in (text, *JSON_OBJECT.findall(text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def text_list(value: object, limit: int | None = None) -> list[str]:
    items = (
        [item.strip() for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []
    )
    return items[:limit]


def select_sources(groups: list[list[Source]], max_sources: int, max_per_article: int | None) -> list[Source]:
    """Choisit les passages à tour de rôle dans chaque groupe de résultats, sans doublon.

    Chaque recherche contribue ainsi aux passages retenus, et un même article ne peut
    occuper plus de `max_per_article` places.
    """
    selected: list[Source] = []
    seen: set[str] = set()
    per_article: Counter[tuple[str, int]] = Counter()
    for rank in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if rank >= len(group) or len(selected) == max_sources:
                continue
            source = group[rank]
            article = (source.document, source.article)
            if source.node_id in seen or (max_per_article and per_article[article] >= max_per_article):
                continue
            seen.add(source.node_id)
            per_article[article] += 1
            selected.append(source)
    return [replace(source, number=number) for number, source in enumerate(selected, start=1)]


def source_names(sources: list[Source]) -> list[str]:
    return [f"[{source.number}] {source.document_name} art. {source.article}" for source in sources]


class Researcher:
    """Décompose la question, lance une recherche par sous-question et sélectionne les passages."""

    def __init__(self, retriever: BaseRetriever, llm: LLM, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    def search(self, query: str) -> list[Source]:
        return [Source.from_node(0, result) for result in self.retriever.retrieve(query)]

    def __call__(self, state: AgentState) -> dict:
        question = state["question"]
        calls = 0
        if state.get("searches"):
            sub_questions = state.get("sub_questions", [question])
            queries = state["searches"]
            groups = [state["sources"]]
        else:
            data = parse_json(chat(self.llm, RESEARCHER_PROMPT, f"Question : {question}"))
            calls = 1
            sub_questions = text_list(data.get("sous_questions"), limit=3) or [question]
            queries = [question] + [sub for sub in sub_questions if sub != question]
            groups = []
        groups += [self.search(query) for query in queries]
        max_per_article = self.settings.max_chunks_per_article if len(groups) > 1 else None
        sources = select_sources(groups, self.settings.max_sources, max_per_article)
        return {
            "sub_questions": sub_questions,
            "queries": queries,
            "sources": sources,
            "searches": [],
            "llm_calls": state.get("llm_calls", 0) + calls,
            "trace": [{"agent": "chercheur", "requetes": queries, "passages": source_names(sources)}],
        }


class Writer:
    """Rédige la réponse à partir des passages, en tenant compte des remarques du vérificateur."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        prompt = build_user_prompt(state["question"], state["sources"])
        sub_questions = state.get("sub_questions", [])
        if len(sub_questions) > 1:
            aspects = "\n".join(f"- {sub}" for sub in sub_questions)
            prompt += "\n\n" + COVERAGE_INSTRUCTIONS.format(aspects=aspects)
        if state.get("problems"):
            problems = "\n".join(f"- {problem}" for problem in state["problems"])
            prompt += "\n\n" + REVISION_INSTRUCTIONS.format(draft=state["draft"], problems=problems)
        draft = chat(self.llm, SYSTEM_PROMPT, prompt)
        update = {
            "draft": draft,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "trace": [{"agent": "rédacteur", "reponse": draft}],
        }
        if "first_draft" not in state:
            update.update(first_draft=draft, first_sources=state["sources"])
        return update


class Verifier:
    """Contrôle que chaque affirmation est justifiée par l'extrait qu'elle cite."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        draft, sources = state["draft"], state["sources"]
        if is_refused_answer(draft, len(sources)):
            return {"verdict": "refus", "problems": [], "trace": [{"agent": "vérificateur", "verdict": "refus"}]}

        prompt = (
            f"Extraits :\n\n{format_context(sources)}\n\n"
            f"Question : {state['question']}\n\nRéponse à vérifier :\n{draft}"
        )
        data = parse_json(chat(self.llm, VERIFIER_PROMPT, prompt))
        verdict = str(data.get("verdict", "")).strip().lower()
        problems = text_list(data.get("problemes"))
        searches = text_list(data.get("recherches"), limit=2) if verdict == "incomplete" else []
        if verdict not in VERDICTS:
            verdict = "illisible"
        if not parse_citations(draft, len(sources)):
            verdict = "a_corriger"
            problems.append("La réponse ne cite aucun extrait.")
        revisions = state.get("revisions", 0) + (verdict in ("a_corriger", "incomplete"))
        return {
            "verdict": verdict,
            "problems": problems,
            "searches": searches,
            "revisions": revisions,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "trace": [{"agent": "vérificateur", "verdict": verdict, "problemes": problems, "recherches": searches}],
        }


def build_graph(researcher: Researcher, writer: Writer, verifier: Verifier | None, max_revisions: int):
    """Chercheur → rédacteur → vérificateur, avec retour au rédacteur ou au chercheur si nécessaire.

    Sans vérificateur, la réponse du rédacteur est renvoyée directement.
    """

    def route(state: AgentState) -> str:
        if state["revisions"] > max_revisions:
            return END
        if state["verdict"] == "incomplete" and state.get("searches"):
            return "chercheur"
        if state["verdict"] in ("a_corriger", "incomplete"):
            return "redacteur"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("chercheur", researcher)
    graph.add_node("redacteur", writer)
    graph.add_edge(START, "chercheur")
    graph.add_edge("chercheur", "redacteur")
    if verifier is None:
        graph.add_edge("redacteur", END)
    else:
        graph.add_node("verificateur", verifier)
        graph.add_edge("redacteur", "verificateur")
        graph.add_conditional_edges("verificateur", route, ["chercheur", "redacteur", END])
    return graph.compile()


class AgentsRAG:
    """Système multi-agents avec la même interface que le RAG simple.

    Si le vérificateur rejette encore la réponse après le nombre maximal de corrections,
    la première réponse est conservée et marquée comme non vérifiée : des corrections
    qui ne convergent pas ne garantissent pas une meilleure réponse.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        json_llm: LLM,
        settings: Settings,
        verifier_llm: LLM | None = None,
        use_verifier: bool = True,
    ):
        self.max_revisions = settings.max_revisions
        self.use_verifier = use_verifier
        verifier = Verifier(verifier_llm or json_llm) if use_verifier else None
        self.graph = build_graph(
            Researcher(retriever, json_llm, settings), Writer(llm), verifier, settings.max_revisions
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None, use_verifier: bool = True) -> "AgentsRAG":
        settings = settings or Settings()
        retriever = get_index(settings).as_retriever(similarity_top_k=settings.top_k)
        json_llm = build_llm(settings, json_mode=True)
        verifier_llm = json_llm
        if settings.verifier_model:
            verifier_llm = build_llm(replace(settings, llm_model=settings.verifier_model), json_mode=True)
        return cls(retriever, build_llm(settings), json_llm, settings, verifier_llm, use_verifier)

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        state = self.graph.invoke({"question": question, "revisions": 0, "llm_calls": 0, "trace": []})
        text, sources = state["draft"], state["sources"]
        if state["revisions"] > self.max_revisions:
            text, sources = state["first_draft"], state["first_sources"]
        refused = is_refused_answer(text, len(sources))
        return Answer(
            question=question,
            text=text,
            sources=sources,
            cited=[] if refused else parse_citations(text, len(sources)),
            refused=refused,
            latency=time.perf_counter() - start,
            llm_calls=state["llm_calls"],
            verified=state["verdict"] in ("valide", "refus") if self.use_verifier else None,
            trace=state["trace"],
        )
