"""API HTTP de l'assistant : une question en entrée, une réponse citée en sortie."""

from collections.abc import Callable
from enum import StrEnum
from functools import partial
from threading import Lock

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from assistant_rag.agents.graph import AgentsRAG
from assistant_rag.config import Settings
from assistant_rag.rag import Answer, SimpleRAG, Source

EXCERPT_LENGTH = 400


class Mode(StrEnum):
    simple = "simple"
    agents = "agents"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, description="Question en français sur le RGPD ou l'AI Act")
    mode: Mode = Mode.simple


class SourceOut(BaseModel):
    number: int
    document: str
    article: int
    title: str
    pages: str
    label: str
    excerpt: str
    cited: bool


class AskResponse(BaseModel):
    question: str
    mode: Mode
    answer: str
    refused: bool
    verified: bool | None
    sources: list[SourceOut]
    latency: float
    llm_calls: int
    trace: list[dict]


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    llm_model: str
    embed_model: str
    index_ready: bool


def source_out(source: Source, cited: set[int]) -> SourceOut:
    excerpt = source.text.split("\n", 1)[-1].strip()
    if len(excerpt) > EXCERPT_LENGTH:
        excerpt = excerpt[:EXCERPT_LENGTH].rsplit(" ", 1)[0] + " …"
    return SourceOut(
        number=source.number,
        document=source.document_name,
        article=source.article,
        title=source.title,
        pages=source.pages,
        label=source.label,
        excerpt=excerpt,
        cited=source.number in cited,
    )


def answer_out(answer: Answer, mode: Mode) -> AskResponse:
    cited = set(answer.cited)
    return AskResponse(
        question=answer.question,
        mode=mode,
        answer=answer.text,
        refused=answer.refused,
        verified=answer.verified,
        sources=[source_out(source, cited) for source in answer.sources],
        latency=round(answer.latency, 2),
        llm_calls=answer.llm_calls,
        trace=answer.trace,
    )


def ollama_available(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/api/tags", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


def default_factories(settings: Settings) -> dict[Mode, Callable[[], object]]:
    return {
        Mode.simple: partial(SimpleRAG.from_settings, settings),
        Mode.agents: partial(AgentsRAG.from_settings, settings),
    }


def create_app(
    settings: Settings | None = None,
    factories: dict[Mode, Callable[[], object]] | None = None,
    ollama_check: Callable[[str], bool] = ollama_available,
) -> FastAPI:
    """Crée l'application. Chaque système est construit au premier appel puis réutilisé."""
    settings = settings or Settings()
    factories = factories or default_factories(settings)
    systems: dict[Mode, object] = {}
    lock = Lock()

    app = FastAPI(
        title="Assistant RAG multi-agents",
        description="Questions-réponses sur le RGPD et l'AI Act, avec citation des sources.",
        version="1.0.0",
    )

    def get_system(mode: Mode):
        with lock:
            if mode not in systems:
                systems[mode] = factories[mode]()
            return systems[mode]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            ollama=ollama_check(settings.ollama_base_url),
            llm_model=settings.llm_model,
            embed_model=settings.embed_model,
            index_ready=settings.index_dir.exists(),
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            answer = get_system(request.mode).answer(request.question.strip())
        except (ConnectionError, httpx.ConnectError) as error:
            detail = "Ollama est injoignable : vérifiez qu'il est démarré."
            raise HTTPException(status_code=503, detail=detail) from error
        return answer_out(answer, request.mode)

    return app


app = create_app()
