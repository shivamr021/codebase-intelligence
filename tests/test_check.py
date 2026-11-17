"""tests/test_check.py — B4b: honest /check (seam B) on seeded harness stores."""
import hashlib
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models as qm

SHA, OLD_VEC = "a" * 40, "fast-bge-small-en"
TWO = {"app.py": "1" * 64, "util.py": "2" * 64}
THREE = {**TWO, "db.py": "3" * 64}
DROP = object()  # seed(..., index_version=DROP) stores metadata without that field

def seed(h, vector=None, meta=True, hashes=TWO, **extra):
    # Stored index for "demo": vector=None -> the app's own embed_and_store (current name); else a raw old collection.
    if vector:
        h.store.create_collection("demo", vectors_config={vector: qm.VectorParams(size=384, distance=qm.Distance.COSINE)})
        h.store.upsert("demo", [qm.PointStruct(id=i, vector={vector: [0.1] * 384}, payload=c) for i, c in enumerate(h.chunks)])
    else:
        from core.embeddings.embedder import embed_and_store
        embed_and_store(h.chunks, "demo", commit_hash=SHA)
    if meta:
        from core.storage.repo_metadata import ensure_metadata_collection
        ensure_metadata_collection()
        payload = {"repo_name": meta if isinstance(meta, str) else "demo", "summary": "s", "mermaid": "flowchart TD\nA-->B",
                   "graph_stats": {"nodes": 2}, "graph_data": {"nodes": ["app.py", "util.py"], "edges": []},
                   "architecture_context": None, "commit_sha": SHA, "file_hashes": hashes, "index_version": 2, **extra}
        payload = {k: v for k, v in payload.items() if v is not DROP}
        h.store.upsert("repo_metadata", [qm.PointStruct(id=hashlib.md5(payload["repo_name"].encode()).hexdigest(),
                                                        vector=[0.0], payload=payload)])

def check(fields=("indexed", "compatible", "files_indexed", "graph_ready")):
    from main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/api/v1/check", params={"repo_name": "demo"})
    body = r.json() if r.headers["content-type"] == "application/json" else {}  # a 500 is plain text
    return r.status_code, *(body.get(f) for f in fields)

@pytest.mark.parametrize("hashes, files", [(TWO, 2), (THREE, 3)], ids=["two-files", "three-files"])
def test_up_to_date_repo_is_compatible_with_stored_file_count_and_graph(harness, hashes, files):
    # Break: compatible/files_indexed/graph_ready missing or hardcoded -> None, or the wrong file count.
    seed(harness, hashes=hashes)
    assert check() == (200, True, True, files, True)

@pytest.mark.parametrize("stale", [{"vector": OLD_VEC}, {"index_version": 1}, {"index_version": DROP}],
                         ids=["old-vector-name", "index-version-1", "index-version-missing"])
def test_stale_index_reports_not_compatible(harness, stale):
    # Break: compatible ignores the stored vector name (G2) or index_version (G3) -> True or missing.
    seed(harness, **stale)
    assert check(("indexed", "compatible")) == (200, True, False)

@pytest.mark.parametrize("meta", [False, "other-repo"], ids=["no-metadata-collection", "only-another-repos-metadata"])
def test_collection_without_metadata_reports_no_files_and_no_graph(harness, meta):
    # Break: files_indexed/graph_ready read from another repo's row, defaulted to True, or missing.
    seed(harness, meta=meta, hashes=THREE)
    assert check(("indexed", "files_indexed", "graph_ready")) == (200, True, 0, False)

def test_failed_architecture_with_saved_graph_still_reports_graph_ready(harness):
    # Break: graph_ready taken from architecture_ok (G6) instead of the saved graph_data -> False.
    seed(harness, architecture_ok=False, summary=DROP, mermaid=DROP)
    assert check(("indexed", "graph_ready")) == (200, True, True)

DEAD = "http://127.0.0.1:9"  # nothing listens here: every Qdrant call fails at once with connection refused

def test_unreachable_qdrant_reports_not_indexed_instead_of_500(harness):
    # Break: the collection lookup error at check.py:52 re-raised -> 500.
    harness.patch("api.routes.check", QdrantClient(url=DEAD, timeout=1))
    assert check(("indexed", "chunks")) == (200, False, 0)

def test_failing_metadata_read_reports_no_files_and_no_graph_instead_of_500(harness):
    # Break: get_repo_metadata at check.py:70 called outside any try (I1) -> 500.
    seed(harness)
    harness.patch("core.storage.repo_metadata", QdrantClient(url=DEAD, timeout=1))
    assert check(("indexed", "files_indexed", "graph_ready")) == (200, True, 0, False)
