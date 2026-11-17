"""tests/test_repo_metadata.py — seam C: repo metadata on the one harness store (B1a middle, B1b edges)."""
import pytest
from fastapi.testclient import TestClient
from core.storage.repo_metadata import get_repo_metadata, save_repo_metadata

def test_one_store_is_seen_by_repo_metadata_check_embedder_and_retriever(harness):
    # Break: any of the 4 module clients unpatched, or a metadata read the store rejects -> /check misreports.
    from core.embeddings.embedder import embed_and_store
    from core.retrieval.retriever import retrieve_chunks
    from main import app
    assert embed_and_store(harness.chunks, "demo")["chunks_stored"] == 2
    save_repo_metadata("demo", "summary", "graph TD", {}, {}, commit_sha="abc123")
    with TestClient(app) as client:
        body = client.get("/api/v1/check", params={"repo_name": "demo"}).json()
    assert (body["indexed"], body["chunks"], body["commit_sha"]) == (True, 2, "abc123")
    assert sorted(c["name"] for c in retrieve_chunks("alpha", "demo")) == ["alpha", "beta"]

def test_missing_metadata_collection_returns_none(harness):
    # Break: first run on an empty Qdrant raises "Collection repo_metadata not found" -> /diagram, /graph 500.
    assert get_repo_metadata("demo") is None

@pytest.mark.parametrize("repo, want", [
    pytest.param("ghost", None, id="edge-repo-absent-returns-none"),
    pytest.param("demo", ("demo summary", {"nodes": 3}, "abc123"), id="edge-repo-present-own-payload"),
    pytest.param("other", ("other summary", {"nodes": 7}, "fff999"), id="edge-repo-present-second-repo"),
])
def test_metadata_lookup_returns_only_the_named_repo(harness, repo, want):
    # Break: no repo_name filter, empty check dropped, or {} returned -> wrong repo's payload, IndexError, {}.
    save_repo_metadata("demo", "demo summary", "graph TD", {"nodes": 3}, {}, commit_sha="abc123")
    save_repo_metadata("other", "other summary", "graph LR", {"nodes": 7}, {}, commit_sha="fff999")
    got = get_repo_metadata(repo)
    assert (got if want is None else (got["summary"], got["graph_stats"], got["commit_sha"])) == want
