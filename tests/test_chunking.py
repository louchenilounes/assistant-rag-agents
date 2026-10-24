from llama_index.core.node_parser import SentenceSplitter

from assistant_rag.chunking import article_header, article_to_chunks, build_chunks
from assistant_rag.extraction import Article


def make_article(number: int = 17, text: str = "Texte court de l'article.") -> Article:
    return Article(
        document="rgpd",
        number=number,
        title="Droit à l'effacement",
        chapter="CHAPITRE III",
        start_page=43,
        end_page=44,
        text=text,
    )


def test_article_header_names_document_and_article():
    assert article_header(make_article()) == "RGPD — Article 17 : Droit à l'effacement"


def test_short_article_gives_one_chunk_with_metadata():
    (node,) = article_to_chunks(make_article(), SentenceSplitter(chunk_size=256, chunk_overlap=20))

    assert node.node_id == "rgpd-art017-00"
    assert node.text.startswith("RGPD — Article 17 : Droit à l'effacement\n")
    assert node.metadata["article"] == 17
    assert node.metadata["start_page"] == 43
    assert node.metadata["chunk_count"] == 1


def test_long_article_is_split_and_each_chunk_keeps_header():
    long_text = " ".join(f"Phrase numéro {index} de l'article sur l'effacement des données." for index in range(200))

    nodes = article_to_chunks(make_article(text=long_text), SentenceSplitter(chunk_size=128, chunk_overlap=16))

    assert len(nodes) > 1
    assert all(node.text.startswith("RGPD — Article 17") for node in nodes)
    assert [node.metadata["chunk"] for node in nodes] == list(range(len(nodes)))


def test_metadata_is_excluded_from_embedded_text():
    (node,) = article_to_chunks(make_article(), SentenceSplitter(chunk_size=256, chunk_overlap=20))

    assert node.get_content(metadata_mode="embed") == node.text
    assert node.get_content(metadata_mode="llm") == node.text


def test_build_chunks_gives_unique_ids():
    nodes = build_chunks([make_article(1), make_article(2), make_article(3)], chunk_size=256, chunk_overlap=20)

    assert len({node.node_id for node in nodes}) == len(nodes) == 3
