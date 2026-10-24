"""Configuration centrale du projet : chemins, documents sources et paramètres du RAG."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STORAGE_DIR = ROOT_DIR / "storage"
RESULTS_DIR = ROOT_DIR / "results"
QUESTIONS_PATH = DATA_DIR / "eval" / "questions.jsonl"
MLFLOW_DB = ROOT_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT_DIR / "mlartifacts"
MLFLOW_EXPERIMENT = "assistant-rag"
INDEX_VERSION = 2


@dataclass(frozen=True)
class SourceDocument:
    """Document du corpus et ses métadonnées de référence."""

    key: str
    name: str
    reference: str
    filename: str
    url: str
    expected_articles: int

    @property
    def path(self) -> Path:
        return RAW_DIR / self.filename


DOCUMENTS: dict[str, SourceDocument] = {
    "rgpd": SourceDocument(
        key="rgpd",
        name="RGPD",
        reference="Règlement (UE) 2016/679",
        filename="rgpd.pdf",
        url="https://eur-lex.europa.eu/legal-content/FR/TXT/PDF/?uri=CELEX:32016R0679",
        expected_articles=99,
    ),
    "ai_act": SourceDocument(
        key="ai_act",
        name="AI Act",
        reference="Règlement (UE) 2024/1689",
        filename="ai_act.pdf",
        url="https://eur-lex.europa.eu/legal-content/FR/TXT/PDF/?uri=CELEX:32024R1689",
        expected_articles=113,
    ),
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    """Paramètres du RAG, modifiables par variables d'environnement."""

    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3:4b-instruct"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "bge-m3"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "qwen3:8b"))
    verifier_model: str = field(default_factory=lambda: _env("VERIFIER_MODEL", ""))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "64")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    temperature: float = 0.0
    context_window: int = field(default_factory=lambda: int(_env("CONTEXT_WINDOW", "6144")))
    max_new_tokens: int = 512
    request_timeout: float = 600.0
    embed_batch_size: int = 16
    max_sources: int = 5
    max_chunks_per_article: int = 2
    max_revisions: int = 2

    @property
    def index_dir(self) -> Path:
        model = self.embed_model.replace(":", "-").replace("/", "-")
        return STORAGE_DIR / f"{model}_c{self.chunk_size}_o{self.chunk_overlap}_v{INDEX_VERSION}"
