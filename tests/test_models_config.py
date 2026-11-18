"""tests/test_models_config.py — B1b: model names and token limit from env (seams E, A, D)."""
import pytest
from qdrant_client import models as qm

@pytest.mark.parametrize("env, sent", [
    pytest.param({}, ("openai/gpt-oss-120b", 8192), id="middle-defaults"),
    pytest.param({"GROQ_MODEL": "openai/gpt-oss-20b"}, ("openai/gpt-oss-20b", 8192), id="middle-model-env"),
    pytest.param({"GROQ_MAX_TOKENS": "4096"}, ("openai/gpt-oss-120b", 4096), id="middle-tokens-env"),
    pytest.param({"GROQ_MODEL": ""}, ("openai/gpt-oss-120b", 8192), id="bad-empty-model-falls-back"),
])
def test_groq_request_carries_model_and_token_limit_from_env(harness, env, sent):
    # Break: model or max_tokens hardcoded in client.py, or copied from config before the env is read.
    harness.reload("core.llm.client", env=env)
    from core.llm.client import call_llm
    assert (call_llm("explain"), [(r["model"], r["max_tokens"]) for r in harness.groq.requests]) == ("fake reply", [sent])

@pytest.mark.parametrize("value", ["abc", "4k"])
def test_non_numeric_token_limit_fails_startup_naming_the_variable(harness, value):
    # Break: GROQ_MAX_TOKENS never read or never checked -> app starts, or dies with a bare int() error.
    from fastapi.testclient import TestClient
    with pytest.raises(ValueError, match="GROQ_MAX_TOKENS"):
        harness.reload("main", env={"GROQ_MAX_TOKENS": value})
        with TestClient(__import__("main").app):
            pass

@pytest.mark.parametrize("env, model, vector, size", [
    pytest.param({}, "BAAI/bge-small-en-v1.5", "fast-bge-small-en-v1.5", 384, id="middle-default-model"),
    pytest.param({"EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"}, "BAAI/bge-base-en-v1.5", "fast-bge-base-en-v1.5", 768,
                 id="edge-model-env"),
])
def test_import_loads_no_model_then_both_clients_use_the_configured_one(harness, env, model, vector, size):
    # Break: model loaded at import (130 MB download per run), or a client ignores the configured model -> another
    # vector name or size is stored, or the query asks a vector the collection lacks and finds nothing.
    harness.reload("core.embeddings.embedder", "core.retrieval.retriever", env=env)
    assert harness.loads == []
    from core.embeddings.embedder import embed_and_store
    from core.retrieval.retriever import retrieve_chunks
    embed_and_store(harness.chunks, "demo")
    vectors = harness.store.get_collection("demo").config.params.vectors
    assert ({k: v.size for k, v in vectors.items()}, len(retrieve_chunks("alpha", "demo")), harness.loads) == (
        {vector: size}, 2, [model])

def test_first_upsert_into_existing_v15_collection_uses_its_vector_name(harness):
    # Break: vector name taken before the configured model is used -> "fast-bge-small-en" rejected by the collection.
    harness.store.create_collection("demo", vectors_config={
        "fast-bge-small-en-v1.5": qm.VectorParams(size=384, distance=qm.Distance.COSINE)})
    from core.embeddings.embedder import upsert_chunks
    assert upsert_chunks(harness.chunks, "demo") == {"status": "success", "repo_name": "demo", "chunks_stored": 2}
    assert harness.store.count("demo").count == 2

def test_missing_baked_model_cache_fails_startup_and_downloads_nothing(harness, tmp_path):
    # P7-B4 new (ruling 4). Break: startup never checks the cache -> first query silently downloads the 130 MB model.
    from fastapi.testclient import TestClient
    harness.reload("core.embeddings.embedder", "core.retrieval.retriever", "main", env={"EMBED_CACHE_DIR": str(tmp_path)})
    with pytest.raises(Exception) as raised:
        with TestClient(__import__("main").app):
            pass
    text = str(raised.value)
    assert (str(tmp_path) in text, "BAAI/bge-small-en-v1.5" in text, harness.downloads) == (True, True, [])
