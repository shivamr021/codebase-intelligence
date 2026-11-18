"""tests/test_vector_space.py — P7: qdrant-client 1.20.0 + vector_space, dense only (seams A, B); one store, fake model."""
import pytest

@pytest.mark.parametrize("env, threads", [
    pytest.param({}, 2, id="edge-threads-default"),
    pytest.param({"EMBED_THREADS": "1"}, 1, id="edge-threads-env"),
])
def test_model_is_built_with_the_thread_count_from_env(harness, env, threads):
    # P7-E3 new. Break: EMBED_THREADS never read, or never passed to the model -> fastembed picks its own thread count.
    harness.reload("core.embeddings.embedder", env=env)
    from core.embeddings.embedder import embed_and_store
    embed_and_store(harness.chunks, "demo")
    assert harness.inits == [("BAAI/bge-small-en-v1.5", threads)]

def test_unreachable_qdrant_raises_and_ask_answers_503_not_404(harness):
    # P7-B2 new. Break: retriever.py:95-97,143-145 turn a refused connection into [] -> /ask 404 "not indexed".
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import ResponseHandlingException
    harness.patch("core.retrieval.retriever", QdrantClient(url="http://127.0.0.1:1", timeout=1))  # a real refused port
    from core.retrieval.retriever import retrieve_chunks
    with pytest.raises(Exception) as raised:
        retrieve_chunks("alpha", "demo")
    chain = [raised.value, raised.value.__cause__, raised.value.__context__]
    from main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        status = client.post("/api/v1/ask", json={"repo_name": "demo", "question": "where is alpha?"}).status_code
    assert (any(isinstance(e, ResponseHandlingException) for e in chain), status, harness.groq.requests) == (True, 503, [])

@pytest.mark.parametrize("asked, want", [
    pytest.param("a" * 40, [("alpha", "def alpha(): return 0", "a" * 40)], id="middle-commit-a"),
    pytest.param("b" * 40, [("beta", "def beta(): return 1", "b" * 40)], id="middle-commit-b"),
])
def test_commit_filter_returns_only_that_commits_chunks(harness, asked, want):
    # P7-M3 existing. Break: commit filter dropped in the move to the new query call -> both commits' chunks come back.
    from core.embeddings.embedder import embed_and_store, upsert_chunks
    from core.retrieval.retriever import retrieve_chunks
    embed_and_store(harness.chunks[:1], "demo", commit_hash="a" * 40)
    upsert_chunks(harness.chunks[1:], "demo", commit_hash="b" * 40)
    got = retrieve_chunks("alpha", "demo", commit_hash=asked)
    assert [(c["name"], c["text"], c["commit_hash"]) for c in got] == want

def test_index_built_today_counts_as_compatible_in_check(harness):
    # P7-M4 existing. Break: vector name built differently from the stored one -> every repo re-indexed, /check false.
    from fastapi.testclient import TestClient
    from core.embeddings.embedder import embed_and_store
    from core.storage.repo_metadata import get_repo_metadata, is_index_compatible, save_repo_metadata
    from main import app
    embed_and_store(harness.chunks, "demo")
    save_repo_metadata("demo", "summary", "graph TD", {}, {}, commit_sha="abc123", extra={"index_version": 2})
    with TestClient(app) as client:
        body = client.get("/api/v1/check", params={"repo_name": "demo"}).json()
    assert (is_index_compatible("demo", get_repo_metadata("demo")), body["compatible"], body["chunks"]) == (True, True, 2)

def test_collection_written_by_qdrant_client_1_9_1_is_read_by_the_new_query(harness):
    # P7-E4 existing. Break: new query uses another vector name, or reads a key 1.9.1 never wrote -> old index unreadable.
    from qdrant_client import models as qm
    from core.retrieval.retriever import retrieve_chunks
    harness.store.create_collection("demo", vectors_config={
        "fast-bge-small-en-v1.5": qm.VectorParams(size=384, distance=qm.Distance.COSINE)})
    metas = [{"text": "def alpha(): return 0", "file_path": "src/alpha.py", "chunk_type": "function", "name": "alpha",
              "start_line": 3, "end_line": 4, "language": "python", "repo_name": "demo", "commit_hash": "a" * 40},
             {"text": "def beta(): return 1", "file_path": "src/beta.py", "chunk_type": "function", "name": "beta",
              "start_line": 7, "end_line": 9, "language": "python", "repo_name": "demo"}]
    harness.store.upsert("demo", [qm.PointStruct(id=i, vector={"fast-bge-small-en-v1.5": [0.1] * 384},
                                                 payload={"document": m["text"], **m}) for i, m in enumerate(metas)])
    got = sorted((c["name"], c["text"], c["file_path"], c["start_line"], c["end_line"]) for c in retrieve_chunks("x", "demo"))
    assert got == [("alpha", "def alpha(): return 0", "src/alpha.py", 3, 4), ("beta", "def beta(): return 1", "src/beta.py", 7, 9)]

def test_missing_collection_returns_empty_and_ask_answers_404(harness):
    # P7-B1 existing. Break: a missing collection raised as a storage error (503/500) or answered with made-up chunks.
    from fastapi.testclient import TestClient
    from core.retrieval.retriever import retrieve_chunks
    from main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        status = client.post("/api/v1/ask", json={"repo_name": "ghost", "question": "where is alpha?"}).status_code
    assert (retrieve_chunks("alpha", "ghost"), status, harness.groq.requests) == ([], 404, [])

def test_upload_refused_by_qdrant_reports_error_and_stores_nothing(harness):
    # P7-B3 existing. Break: upload failure reported as success -> ingest saves metadata over an empty index.
    from qdrant_client import models as qm
    from core.embeddings.embedder import upsert_chunks
    harness.store.create_collection("demo", vectors_config={
        "fast-bge-small-en-v1.5": qm.VectorParams(size=768, distance=qm.Distance.COSINE)})  # the model makes 384
    got = upsert_chunks(harness.chunks, "demo")
    assert (got["status"], got["chunks_stored"], harness.store.count("demo").count) == ("error", 0, 0)
