"""Interface web de l'assistant : pose une question à l'API et affiche la réponse avec ses sources."""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 900
MODES = {"RAG simple": "simple", "Multi-agents": "agents"}
EXAMPLES = [
    "Quel est le délai pour notifier une violation de données à l'autorité de contrôle ?",
    "Quelles pratiques d'intelligence artificielle sont interdites par l'AI Act ?",
    "Qu'est-ce qu'un délégué à la protection des données et quand doit-il être désigné ?",
]
VERDICTS = {
    "valide": "validée",
    "a_corriger": "à corriger",
    "incomplete": "incomplète",
    "illisible": "illisible",
    "refus": "refus",
}

st.set_page_config(page_title="Assistant RGPD & AI Act", page_icon="⚖️", layout="wide")


def get_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def ask(question: str, mode: str) -> dict:
    response = requests.post(
        f"{API_URL}/ask", json={"question": question, "mode": mode}, timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 200:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(detail if isinstance(detail, str) else "Question invalide.")
    return response.json()


def render_sources(sources: list[dict]) -> None:
    for source in sources:
        st.markdown(f"**[{source['number']}] {source['label']}**")
        st.caption(source["excerpt"])


def render_trace(trace: list[dict]) -> None:
    for step in trace:
        agent = step["agent"].capitalize()
        if "requetes" in step:
            st.markdown(f"**{agent}** : {len(step['requetes'])} recherche(s)")
            for query in step["requetes"]:
                st.markdown(f"- {query}")
        elif "reponse" in step:
            st.markdown(f"**{agent}** : brouillon rédigé")
        elif "verdict" in step:
            st.markdown(f"**{agent}** : réponse {VERDICTS.get(step['verdict'], step['verdict'])}")
            for problem in step.get("problemes", []):
                st.markdown(f"- {problem}")


def render_answer(result: dict) -> None:
    st.markdown(result["answer"])
    if result["verified"] is False:
        st.warning("Réponse non validée par le vérificateur : à contrôler dans les sources.")

    cited = [source for source in result["sources"] if source["cited"]]
    others = [source for source in result["sources"] if not source["cited"]]
    if cited:
        with st.expander(f"Sources citées ({len(cited)})", expanded=True):
            render_sources(cited)
    if others:
        with st.expander(f"Autres passages consultés ({len(others)})"):
            render_sources(others)
    if result["trace"]:
        with st.expander("Étapes des agents"):
            render_trace(result["trace"])
    st.caption(f"{result['latency']:.1f} s · {result['llm_calls']} appel(s) au LLM · mode {result['mode']}")


with st.sidebar:
    st.header("Paramètres")
    mode_label = st.radio("Système", list(MODES), help="Le RAG simple est le plus fiable et le plus rapide.")
    health = get_health()
    if health is None:
        st.error(f"API injoignable ({API_URL})")
    else:
        if health["ollama"]:
            st.success("API et Ollama disponibles")
        else:
            st.warning("Ollama ne répond pas")
        if not health["index_ready"]:
            st.warning("Index absent : il sera construit à la première question.")
        st.caption(f"LLM : {health['llm_model']}  \nEmbeddings : {health['embed_model']}")
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
    st.divider()
    st.caption("Corpus : RGPD (Règlement (UE) 2016/679) et AI Act (Règlement (UE) 2024/1689).")

st.title("Assistant RGPD & AI Act")
st.write("Posez une question sur le RGPD ou l'AI Act : chaque réponse cite les articles sur lesquels elle s'appuie.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Exemples de questions**")
    for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
        if column.button(example, use_container_width=True):
            st.session_state.pending = example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_answer(message["content"])

question = st.chat_input("Votre question") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents et rédaction de la réponse…"):
            try:
                result = ask(question, MODES[mode_label])
            except (requests.RequestException, RuntimeError) as error:
                st.error(f"Impossible d'obtenir une réponse : {error}")
            else:
                render_answer(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
