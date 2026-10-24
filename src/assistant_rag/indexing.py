"""Construction, sauvegarde et chargement de l'index vectoriel."""

from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding

from assistant_rag.chunking import build_chunks
from assistant_rag.config import Settings
from assistant_rag.extraction import load_corpus, split_articles


def build_embed_model(settings: Settings) -> BaseEmbedding:
    return OllamaEmbedding(
        model_name=settings.embed_model,
        base_url=settings.ollama_base_url,
        embed_batch_size=settings.embed_batch_size,
    )


def load_corpus_chunks(settings: Settings) -> list[TextNode]:
    """Charge les documents, les découpe en articles puis en chunks."""
    articles = [article for pages in load_corpus().values() for article in split_articles(pages)]
    return build_chunks(articles, settings.chunk_size, settings.chunk_overlap)


def build_index(
    nodes: list[TextNode], embed_model: BaseEmbedding, persist_dir: Path, show_progress: bool = True
) -> VectorStoreIndex:
    """Calcule les embeddings des chunks et sauvegarde l'index sur disque."""
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=show_progress)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return index


def load_index(persist_dir: Path, embed_model: BaseEmbedding) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_context, embed_model=embed_model)


def get_index(settings: Settings, embed_model: BaseEmbedding | None = None, rebuild: bool = False) -> VectorStoreIndex:
    """Charge l'index correspondant à la configuration, ou le construit s'il n'existe pas."""
    embed_model = embed_model or build_embed_model(settings)
    persist_dir = settings.index_dir
    if persist_dir.exists() and not rebuild:
        return load_index(persist_dir, embed_model)
    return build_index(load_corpus_chunks(settings), embed_model, persist_dir)
