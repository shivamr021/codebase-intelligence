"""tests/test_harness.py — declared guards for the conftest.py harness (B1a): they pass once it exists."""
import pytest
from fastapi.testclient import TestClient

def test_app_starts_with_dummy_env(harness):
    # Break: harness skips the dummy env -> validate_config raises at startup and every TestClient test errors.
    from main import app
    with TestClient(app) as client:
        assert client.get("/").json()["status"] == "running"

def test_model_loader_is_faked_so_embedding_needs_no_download(harness):
    # Break: harness leaves fastembed's real TextEmbedding -> download over a blocked network, 0 chunks stored.
    from core.embeddings.embedder import embed_and_store
    stored = embed_and_store(harness.chunks, "demo")["chunks_stored"]
    assert (stored, harness.store.count("demo").count, len(harness.loads)) == (2, 2, 1)

@pytest.mark.parametrize("shell_env", [{"GROQ_API_KEY": "gsk_real_looking_key", "QDRANT_URL": "https://real.qdrant.io"}])
def test_real_shell_keys_are_overridden_and_never_reach_groq(harness):
    # Break: shell env applied after the dummy env, or the Groq client unpatched -> live Groq/Qdrant Cloud.
    import config
    from core.llm.client import call_llm
    assert (config.GROQ_API_KEY, config.QDRANT_URL) == ("test-dummy-groq-key", "http://qdrant.invalid:6333")
    assert (call_llm("hi"), len(harness.groq.requests)) == ("fake reply", 1)
