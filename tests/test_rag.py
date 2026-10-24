from typing import Any

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

from assistant_rag.chunking import build_chunks
from assistant_rag.citations import REFUSAL
from assistant_rag.extraction import Article
from assistant_rag.indexing import build_index, load_index
from assistant_rag.rag import SimpleRAG, build_user_prompt


class ScriptedLLM(CustomLLM):
    reply: str = ""
    prompts: list[str] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        return CompletionResponse(text=self.reply)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        yield self.complete(prompt)


ARTICLES = [
    Article("rgpd", 17, "Droit à l'effacement", "CHAPITRE III", 43, 44, "La personne concernée a le droit d'obtenir."),
    Article("rgpd", 37, "Désignation du délégué", "CHAPITRE IV", 55, 55, "Le responsable du traitement désigne."),
    Article("ai_act", 5, "Pratiques interdites", "CHAPITRE II", 51, 53, "Les pratiques suivantes sont interdites."),
]


@pytest.fixture
def index(tmp_path):
    nodes = build_chunks(ARTICLES, chunk_size=256, chunk_overlap=20)
    return build_index(nodes, MockEmbedding(embed_dim=8), tmp_path / "index", show_progress=False)


def make_rag(index, reply: str) -> SimpleRAG:
    return SimpleRAG(index.as_retriever(similarity_top_k=3), ScriptedLLM(reply=reply, prompts=[]))


def test_index_is_persisted_and_reloaded(index, tmp_path):
    reloaded = load_index(tmp_path / "index", MockEmbedding(embed_dim=8))

    assert set(reloaded.docstore.docs) == {"rgpd-art017-00", "rgpd-art037-00", "ai_act-art005-00"}


def test_retrieve_numbers_sources_with_citation_metadata(index):
    sources = make_rag(index, "").retrieve("effacement")

    assert [source.number for source in sources] == [1, 2, 3]
    assert {source.label for source in sources} == {
        "RGPD, article 17 « Droit à l'effacement », p. 43-44",
        "RGPD, article 37 « Désignation du délégué », p. 55",
        "AI Act, article 5 « Pratiques interdites », p. 51-53",
    }


def test_prompt_contains_numbered_extracts_and_question(index):
    rag = make_rag(index, "")
    sources = rag.retrieve("effacement")

    prompt = build_user_prompt("Qu'est-ce que le droit à l'effacement ?", sources)

    assert prompt.startswith("Extraits :\n\n[1] ")
    assert "[3] " in prompt
    assert prompt.endswith("Question : Qu'est-ce que le droit à l'effacement ?")


def test_answer_maps_citations_to_sources(index):
    answer = make_rag(index, "<think>brouillon</think>Le droit existe [2]. Il est encadré [2, 3] [7].").answer("q")

    assert answer.text == "Le droit existe [2]. Il est encadré [2, 3] [7]."
    assert answer.cited == [2, 3]
    assert [source.number for source in answer.cited_sources] == [2, 3]
    assert not answer.refused
    assert answer.format().splitlines()[-2:] == [
        f"[2] {answer.sources[1].label}",
        f"[3] {answer.sources[2].label}",
    ]


def test_refusal_has_no_cited_sources(index):
    answer = make_rag(index, REFUSAL).answer("Quelle est la durée légale du travail ?")

    assert answer.refused
    assert answer.cited == []
    assert answer.format() == REFUSAL


def test_llm_receives_system_rules_and_context(index):
    rag = make_rag(index, "Réponse [1].")

    rag.answer("effacement")

    prompt = rag.llm.prompts[-1]
    assert "uniquement à partir des extraits" in prompt
    assert "Question : effacement" in prompt
