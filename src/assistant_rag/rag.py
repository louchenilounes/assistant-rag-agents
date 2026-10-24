"""RAG simple : recherche des passages pertinents puis génération d'une réponse citée."""

import time
from dataclasses import dataclass, field

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama

from assistant_rag.citations import is_refusal, parse_citations, strip_thinking
from assistant_rag.config import Settings
from assistant_rag.indexing import get_index
from assistant_rag.prompts import SYSTEM_PROMPT


@dataclass(frozen=True)
class Source:
    """Passage retrouvé, numéroté tel qu'il est présenté au LLM."""

    number: int
    node_id: str
    document: str
    document_name: str
    article: int
    title: str
    start_page: int
    end_page: int
    text: str
    score: float

    @property
    def pages(self) -> str:
        if self.start_page == self.end_page:
            return f"p. {self.start_page}"
        return f"p. {self.start_page}-{self.end_page}"

    @property
    def label(self) -> str:
        return f"{self.document_name}, article {self.article} « {self.title} », {self.pages}"

    @classmethod
    def from_node(cls, number: int, result: NodeWithScore) -> "Source":
        metadata = result.node.metadata
        return cls(
            number=number,
            node_id=result.node.node_id,
            document=metadata["document"],
            document_name=metadata["document_name"],
            article=int(metadata["article"]),
            title=metadata["title"],
            start_page=int(metadata["start_page"]),
            end_page=int(metadata["end_page"]),
            text=result.node.get_content(),
            score=float(result.score or 0.0),
        )


@dataclass
class Answer:
    """Réponse générée avec ses sources."""

    question: str
    text: str
    sources: list[Source]
    cited: list[int] = field(default_factory=list)
    refused: bool = False
    latency: float = 0.0

    @property
    def cited_sources(self) -> list[Source]:
        return [source for source in self.sources if source.number in self.cited]

    def format(self) -> str:
        """Réponse suivie de ses sources, les passages d'un même article étant regroupés sur une ligne."""
        grouped: dict[str, list[int]] = {}
        for source in self.cited_sources:
            grouped.setdefault(source.label, []).append(source.number)
        lines = [self.text]
        if grouped:
            lines += ["", "Sources :"]
            lines += [f"[{', '.join(map(str, numbers))}] {label}" for label, numbers in grouped.items()]
        return "\n".join(lines)


def format_context(sources: list[Source]) -> str:
    return "\n\n".join(f"[{source.number}] {source.label}\n{source.text}" for source in sources)


def build_user_prompt(question: str, sources: list[Source]) -> str:
    return f"Extraits :\n\n{format_context(sources)}\n\nQuestion : {question}"


def build_llm(settings: Settings) -> LLM:
    return Ollama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        context_window=settings.context_window,
        request_timeout=settings.request_timeout,
        additional_kwargs={"num_predict": settings.max_new_tokens},
        thinking=False,
    )


class SimpleRAG:
    """Recherche les k passages les plus proches de la question puis génère une réponse citée."""

    def __init__(self, retriever: BaseRetriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "SimpleRAG":
        settings = settings or Settings()
        index = get_index(settings)
        return cls(index.as_retriever(similarity_top_k=settings.top_k), build_llm(settings))

    def retrieve(self, question: str) -> list[Source]:
        results = self.retriever.retrieve(question)
        return [Source.from_node(number, result) for number, result in enumerate(results, start=1)]

    def generate(self, question: str, sources: list[Source]) -> str:
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=build_user_prompt(question, sources)),
        ]
        response = self.llm.chat(messages)
        return strip_thinking(response.message.content or "")

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        sources = self.retrieve(question)
        text = self.generate(question, sources)
        refused = is_refusal(text)
        return Answer(
            question=question,
            text=text,
            sources=sources,
            cited=[] if refused else parse_citations(text, len(sources)),
            refused=refused,
            latency=time.perf_counter() - start,
        )
