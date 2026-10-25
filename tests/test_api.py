import pytest
from fastapi.testclient import TestClient

from assistant_rag.api import Mode, create_app
from assistant_rag.config import Settings
from assistant_rag.rag import Answer, Source


def make_source(number: int, text: str = "Article 33\nNotification d'une violation") -> Source:
    return Source(
        number=number,
        node_id=f"rgpd-art033-{number:02d}",
        document="rgpd",
        document_name="RGPD",
        article=33,
        title="Notification à l'autorité de contrôle",
        start_page=52,
        end_page=52,
        text=text,
        score=0.8,
    )


class FakeSystem:
    def __init__(self, verified: bool | None = None, error: Exception | None = None):
        self.verified = verified
        self.error = error
        self.questions: list[str] = []

    def answer(self, question: str) -> Answer:
        if self.error:
            raise self.error
        self.questions.append(question)
        return Answer(
            question=question,
            text="Dans un délai de 72 heures [1].",
            sources=[make_source(1), make_source(2, "Article 33\n" + "mot " * 200)],
            cited=[1],
            latency=1.234,
            llm_calls=1 if self.verified is None else 3,
            verified=self.verified,
            trace=[] if self.verified is None else [{"agent": "vérificateur", "verdict": "valide"}],
        )


@pytest.fixture
def systems():
    return {Mode.simple: FakeSystem(), Mode.agents: FakeSystem(verified=True)}


@pytest.fixture
def client(systems):
    factories = {mode: (lambda system=system: system) for mode, system in systems.items()}
    return TestClient(create_app(Settings(), factories, ollama_check=lambda url: True))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ollama"] is True
    assert body["llm_model"] == Settings().llm_model


def test_ask_simple(client, systems):
    response = client.post("/ask", json={"question": "  Quel délai de notification ?  "})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "simple"
    assert body["answer"].startswith("Dans un délai")
    assert body["verified"] is None
    assert body["latency"] == 1.23
    assert [source["cited"] for source in body["sources"]] == [True, False]
    assert body["sources"][0]["label"] == "RGPD, article 33 « Notification à l'autorité de contrôle », p. 52"
    assert systems[Mode.simple].questions == ["Quel délai de notification ?"]


def test_excerpt_is_truncated(client):
    excerpt = client.post("/ask", json={"question": "Quel délai ?"}).json()["sources"][1]["excerpt"]
    assert excerpt.endswith("…")
    assert len(excerpt) <= 402
    assert not excerpt.startswith("Article 33")


def test_ask_agents(client, systems):
    body = client.post("/ask", json={"question": "Quel délai ?", "mode": "agents"}).json()
    assert body["verified"] is True
    assert body["llm_calls"] == 3
    assert body["trace"][0]["agent"] == "vérificateur"
    assert systems[Mode.simple].questions == []


def test_system_is_built_once():
    calls = []

    def factory():
        calls.append(1)
        return FakeSystem()

    client = TestClient(create_app(Settings(), {Mode.simple: factory}, ollama_check=lambda url: True))
    for _ in range(3):
        client.post("/ask", json={"question": "Quel délai ?"})
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "a" * 1001}, {"question": "Quel délai ?", "mode": "autre"}, {}],
)
def test_invalid_requests(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ollama_unreachable():
    system = FakeSystem(error=ConnectionError("refused"))
    client = TestClient(create_app(Settings(), {Mode.simple: lambda: system}, ollama_check=lambda url: False))
    response = client.post("/ask", json={"question": "Quel délai ?"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]
    assert client.get("/health").json()["ollama"] is False
