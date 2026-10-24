from typing import Any

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

from assistant_rag.agents.graph import AgentsRAG, parse_json, select_sources
from assistant_rag.chunking import build_chunks
from assistant_rag.citations import REFUSAL
from assistant_rag.config import Settings
from assistant_rag.extraction import Article
from assistant_rag.indexing import build_index
from assistant_rag.rag import Source

RESEARCHER = "Tu prépares une recherche documentaire"
VERIFIER = "Tu vérifies une réponse"
WRITER = "Tu es un assistant spécialisé"


class RoleLLM(CustomLLM):
    """Faux LLM qui répond selon l'agent qui l'appelle, avec une file de réponses par agent."""

    replies: dict[str, list[str]] = {}
    prompts: list[str] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        role = next(marker for marker in (RESEARCHER, VERIFIER, WRITER) if marker in prompt)
        queue = self.replies[role]
        return CompletionResponse(text=queue.pop(0) if len(queue) > 1 else queue[0])

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        yield self.complete(prompt)


ARTICLES = [
    Article("rgpd", 33, "Notification à l'autorité", "CHAPITRE IV", 52, 52, "Notification dans les 72 heures."),
    Article("rgpd", 34, "Communication à la personne", "CHAPITRE IV", 53, 53, "Communication en cas de risque élevé."),
    Article("ai_act", 99, "Sanctions", "CHAPITRE XII", 126, 128, "Amendes jusqu'à 35 000 000 EUR."),
]


@pytest.fixture
def retriever(tmp_path):
    nodes = build_chunks(ARTICLES, chunk_size=256, chunk_overlap=20)
    index = build_index(nodes, MockEmbedding(embed_dim=8), tmp_path / "index", show_progress=False)
    return index.as_retriever(similarity_top_k=3)


def make_system(retriever, replies: dict[str, list[str]]) -> tuple[AgentsRAG, RoleLLM]:
    llm = RoleLLM(replies=replies, prompts=[])
    settings = Settings(max_sources=3, max_chunks_per_article=1, max_revisions=2)
    return AgentsRAG(retriever, llm, llm, settings), llm


def source(node_id: str, article: int) -> Source:
    return Source(0, node_id, "rgpd", "RGPD", article, "Titre", 1, 1, "", 0.5)


def test_select_sources_alternates_between_searches_and_limits_each_article():
    first = [source("a-0", 1), source("a-1", 1), source("b-0", 2)]
    second = [source("c-0", 3), source("a-0", 1), source("d-0", 4)]

    selected = select_sources([first, second], max_sources=4, max_per_article=1)

    assert [s.node_id for s in selected] == ["a-0", "c-0", "b-0", "d-0"]
    assert [s.number for s in selected] == [1, 2, 3, 4]


def test_select_sources_without_article_limit_keeps_ranking():
    first = [source("a-0", 1), source("a-1", 1), source("a-2", 1)]

    assert [s.node_id for s in select_sources([first], 2, None)] == ["a-0", "a-1"]


def test_parse_json_reads_object_surrounded_by_text():
    assert parse_json('Voici : {"verdict": "valide"} fin') == {"verdict": "valide"}
    assert parse_json("illisible") == {}


def test_valid_answer_goes_through_the_three_agents_once(retriever):
    system, llm = make_system(
        retriever,
        {
            RESEARCHER: ['{"sous_questions": ["Délai de notification ?"]}'],
            WRITER: ["La notification intervient dans les 72 heures [1]."],
            VERIFIER: ['{"verdict": "valide", "problemes": [], "recherches": []}'],
        },
    )

    answer = system.answer("Délai de notification ?")

    assert [step["agent"] for step in answer.trace] == ["chercheur", "rédacteur", "vérificateur"]
    assert answer.llm_calls == 3
    assert answer.verified is True
    assert answer.cited == [1]


def test_multi_aspect_question_searches_each_sub_question(retriever):
    system, _ = make_system(
        retriever,
        {
            RESEARCHER: ['{"sous_questions": ["Notification à l\'autorité ?", "Communication aux personnes ?"]}'],
            WRITER: ["Réponse [1] [2]."],
            VERIFIER: ['{"verdict": "valide"}'],
        },
    )

    answer = system.answer("Quelles obligations en cas de violation ?")

    queries = answer.trace[0]["requetes"]
    assert queries == [
        "Quelles obligations en cas de violation ?",
        "Notification à l'autorité ?",
        "Communication aux personnes ?",
    ]
    articles = [(s.document, s.article) for s in answer.sources]
    assert len(articles) == len(set(articles))


def test_rejected_answer_is_rewritten_with_verifier_feedback(retriever):
    system, llm = make_system(
        retriever,
        {
            RESEARCHER: ['{"sous_questions": ["Amende ?"]}'],
            WRITER: ["Amende de 1 500 000 EUR [1].", "Amende de 35 000 000 EUR [1]."],
            VERIFIER: [
                '{"verdict": "a_corriger", "problemes": ["Le montant ne figure pas dans l\'extrait 1."]}',
                '{"verdict": "valide"}',
            ],
        },
    )

    answer = system.answer("Amende ?")

    assert answer.text == "Amende de 35 000 000 EUR [1]."
    agents = [step["agent"] for step in answer.trace]
    assert agents == ["chercheur", "rédacteur", "vérificateur", "rédacteur", "vérificateur"]
    revision_prompt = [p for p in llm.prompts if WRITER in p][-1]
    assert "Le montant ne figure pas dans l'extrait 1." in revision_prompt
    assert "Amende de 1 500 000 EUR [1]." in revision_prompt


def test_incomplete_answer_triggers_a_new_search(retriever):
    system, _ = make_system(
        retriever,
        {
            RESEARCHER: ['{"sous_questions": ["Notification ?"]}'],
            WRITER: ["Notification en 72 heures [1]."],
            VERIFIER: [
                '{"verdict": "incomplete", "problemes": ["Les personnes ne sont pas traitées."], '
                '"recherches": ["communication aux personnes concernées"]}',
                '{"verdict": "valide"}',
            ],
        },
    )

    answer = system.answer("Notification et communication ?")

    agents = [step["agent"] for step in answer.trace]
    assert agents == ["chercheur", "rédacteur", "vérificateur", "chercheur", "rédacteur", "vérificateur"]
    assert answer.trace[3]["requetes"] == ["communication aux personnes concernées"]
    assert answer.llm_calls == 5


def test_revisions_stop_after_the_limit(retriever):
    system, _ = make_system(
        retriever,
        {
            RESEARCHER: ['{"sous_questions": ["Amende ?"]}'],
            WRITER: ["Amende [1]."],
            VERIFIER: ['{"verdict": "a_corriger", "problemes": ["Erreur."]}'],
        },
    )

    answer = system.answer("Amende ?")

    assert [step["agent"] for step in answer.trace].count("rédacteur") == 3
    assert answer.verified is False


def test_refusal_is_accepted_without_verification_call(retriever):
    system, _ = make_system(
        retriever,
        {
            RESEARCHER: ['{"sous_questions": ["Durée du travail ?"]}'],
            WRITER: [REFUSAL],
            VERIFIER: ['{"verdict": "valide"}'],
        },
    )

    answer = system.answer("Durée du travail ?")

    assert answer.refused
    assert answer.llm_calls == 2
    assert answer.trace[-1] == {"agent": "vérificateur", "verdict": "refus"}


def test_answer_without_citation_is_sent_back_to_the_writer(retriever):
    system, _ = make_system(
        retriever,
        {
            RESEARCHER: ["pas de JSON"],
            WRITER: ["Réponse sans citation.", "Réponse citée [1]."],
            VERIFIER: ['{"verdict": "valide"}'],
        },
    )

    answer = system.answer("Question ?")

    assert answer.trace[0]["requetes"] == ["Question ?"]
    assert answer.trace[2]["problemes"] == ["La réponse ne cite aucun extrait."]
    assert answer.text == "Réponse citée [1]."
