"""Découpage des articles en chunks indexables, avec leurs métadonnées de citation."""

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from assistant_rag.config import DOCUMENTS
from assistant_rag.extraction import Article

METADATA_KEYS = ["document", "document_name", "reference", "article", "title", "chapter", "start_page", "end_page"]


def article_header(article: Article) -> str:
    """Titre placé en tête de chaque chunk : il situe le passage pour la recherche et pour le LLM."""
    name = DOCUMENTS[article.document].name
    return f"{name} — Article {article.number} : {article.title}"


def article_to_chunks(article: Article, splitter: SentenceSplitter) -> list[TextNode]:
    """Transforme un article en un ou plusieurs chunks.

    Un article court donne un seul chunk ; un article long est redécoupé par
    phrases, chaque morceau recevant le titre de l'article.
    """
    source = DOCUMENTS[article.document]
    header = article_header(article)
    parts = splitter.split_text(article.text) if article.text else [""]
    metadata = {
        "document": article.document,
        "document_name": source.name,
        "reference": source.reference,
        "article": article.number,
        "title": article.title,
        "chapter": article.chapter or "",
        "start_page": article.start_page,
        "end_page": article.end_page,
    }
    nodes = []
    for index, part in enumerate(parts):
        node = TextNode(
            id_=f"{article.document}-art{article.number:03d}-{index:02d}",
            text=f"{header}\n{part}".strip(),
            metadata={**metadata, "chunk": index, "chunk_count": len(parts)},
        )
        node.excluded_embed_metadata_keys = list(node.metadata)
        node.excluded_llm_metadata_keys = list(node.metadata)
        nodes.append(node)
    return nodes


def build_chunks(articles: list[Article], chunk_size: int, chunk_overlap: int) -> list[TextNode]:
    """Découpe une liste d'articles en chunks."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, include_prev_next_rel=False)
    return [node for article in articles for node in article_to_chunks(article, splitter)]
