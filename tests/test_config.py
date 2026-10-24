from assistant_rag.config import STORAGE_DIR, Settings


def test_index_dir_depends_on_embedding_and_chunking():
    settings = Settings(embed_model="bge-m3:latest", chunk_size=256, chunk_overlap=32)

    assert settings.index_dir == STORAGE_DIR / "bge-m3-latest_c256_o32"


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")

    settings = Settings()

    assert (settings.llm_model, settings.top_k, settings.ollama_base_url) == ("qwen3:8b", 8, "http://ollama:11434")
