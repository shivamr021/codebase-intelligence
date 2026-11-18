"""tests/test_models_config.py — B1b: model names and token limit from env (seams E, A, D)."""
import pytest
from qdrant_client import QdrantClient, models as qm

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
    # Break: set_model at import (130 MB download per run), or a client never calls it -> library default bge-small-en.
    harness.reload("core.embeddings.embedder", "core.retrieval.retriever", env=env)
    assert harness.loads == []
    harness.patch("core.embeddings.embedder", emb_store := QdrantClient(":memory:"))  # seam A exception: own stores
    harness.patch("core.retrieval.retriever", ret_store := QdrantClient(":memory:"))
    ret_store.create_collection("demo", vectors_config={vector: qm.VectorParams(size=size, distance=qm.Distance.COSINE)})
    from core.embeddings.embedder import embed_and_store
    from core.retrieval.retriever import retrieve_chunks
    embed_and_store(harness.chunks, "demo")
    retrieve_chunks("alpha", "demo")
    assert (emb_store.embedding_model_name, ret_store.embedding_model_name, harness.loads) == (model, model, [model])

def test_first_upsert_into_existing_v15_collection_uses_its_vector_name(harness):
    # Break: vector name looked up before set_model -> "fast-bge-small-en" rejected by the stored collection.
    harness.store.create_collection("demo", vectors_config={
        "fast-bge-small-en-v1.5": qm.VectorParams(size=384, distance=qm.Distance.COSINE)})
    from core.embeddings.embedder import upsert_chunks
    assert upsert_chunks(harness.chunks, "demo") == {"status": "success", "repo_name": "demo", "chunks_stored": 2}
    assert harness.store.count("demo").count == 2
