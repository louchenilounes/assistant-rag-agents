from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

HEALTH = {
    "status": "ok",
    "ollama": True,
    "llm_model": "qwen3:4b-instruct",
    "embed_model": "bge-m3",
    "index_ready": True,
}
RESULT = {
    "question": "Quel délai ?",
    "mode": "agents",
    "answer": "Dans un délai de 72 heures [1].",
    "refused": False,
    "verified": False,
    "sources": [
        {
            "number": 1,
            "document": "RGPD",
            "article": 33,
            "title": "Notification",
            "pages": "p. 52",
            "label": "RGPD, article 33 « Notification », p. 52",
            "excerpt": "En cas de violation de données…",
            "cited": True,
        }
    ],
    "latency": 12.5,
    "llm_calls": 4,
    "trace": [
        {"agent": "chercheur", "requetes": ["délai notification violation"], "passages": []},
        {"agent": "vérificateur", "verdict": "incomplete", "problemes": ["Délai non précisé"], "recherches": []},
    ],
}


def response(payload: dict, status: int = 200) -> MagicMock:
    mock = MagicMock(status_code=status)
    mock.json.return_value = payload
    return mock


@patch("requests.get", return_value=response(HEALTH))
def test_home_page(_get):
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert app.title[0].value == "Assistant RGPD & AI Act"
    assert app.sidebar.success[0].value == "API et Ollama disponibles"


@patch("requests.get", side_effect=__import__("requests").ConnectionError())
def test_api_unreachable(_get):
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert "API injoignable" in app.sidebar.error[0].value


@patch("requests.post", return_value=response(RESULT))
@patch("requests.get", return_value=response(HEALTH))
def test_question_displays_answer(_get, post):
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    app.sidebar.radio[0].set_value("Multi-agents").run()
    app.chat_input[0].set_value("Quel délai ?").run()
    assert not app.exception
    assert post.call_args.kwargs["json"] == {"question": "Quel délai ?", "mode": "agents"}
    markdown = [element.value for element in app.markdown]
    assert "Dans un délai de 72 heures [1]." in markdown
    assert "non validée" in app.warning[0].value
    assert len(app.session_state.messages) == 2


@patch("requests.post", return_value=response(RESULT))
@patch("requests.get", return_value=response(HEALTH))
def test_example_button_asks_question(_get, post):
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    app.button[1].click().run()
    assert not app.exception
    assert post.call_args.kwargs["json"]["question"].startswith("Quelles pratiques")
    assert "Exemples de questions" not in [element.value for element in app.markdown]
