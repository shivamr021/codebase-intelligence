"""tests/test_safe_rebuild.py — P10 safe rebuild (seam A): /ingest, /ask, /check through the test client; fake model
raising on a chosen text, Qdrant wrapper raising on a chosen call, ingest held at a fake download."""
import fastembed
import pytest
import requests
from fastapi.testclient import TestClient
from test_concurrency import ingest, while_held  # demo's ingest held at its download (test_concurrency.py:19-28)
from test_index_version import stored_version  # raw stored index_version of "o--demo"
from test_temp_folder import FILES_A, FakeGitHub, zip_of

NAME = "o--demo"  # AD8 collection of https://github.com/o/demo

def serve(harness, monkeypatch, tmp_path, files):  # one client; demo's archive serves gh.zips[None]
    harness.reload(env={"TMP_DIR": str(tmp_path)})
    monkeypatch.setattr("core.ingestion.cloner.TMP_DIR", str(tmp_path))
    monkeypatch.setattr(requests, "get", (gh := FakeGitHub({None: zip_of("demo-main", files)})).get)
    from main import app
    return TestClient(app, raise_server_exceptions=False), gh

@pytest.fixture
def world(harness, monkeypatch, tmp_path):
    return lambda files: serve(harness, monkeypatch, tmp_path, files)

def fn(name):  # one function, no top-level code: one chunk
    return f"def {name}():\n    total = sum(len(str(i)) for i in range(9))\n    return total\n\n\n"
MANY = {"mod.py": "".join(fn(f"f{i:03d}") for i in range(105))}  # 105 chunks
FIVE = {"a.py": fn("a1") + fn("a2"), "b.py": fn("b1") + fn("b2"), "c.py": fn("c1")}
OLD_FIVE = [("a.py", "a1"), ("a.py", "a2"), ("b.py", "b1"), ("b.py", "b2"), ("c.py", "c1")]

def points(harness):  # (file_path, name) of every stored point of demo
    found, _ = harness.store.scroll(NAME, limit=500)
    return sorted((p.payload["file_path"], p.payload["name"]) for p in found)

def model_raises_at(monkeypatch, nth):  # the model crashes on the nth text it receives; returns the disarm switch
    seen, cls = [], fastembed.TextEmbedding
    orig = cls.passage_embed
    def embed(model, documents, **kwargs):
        docs = list(documents)
        if len(seen) < nth <= len(seen) + len(docs):
            raise RuntimeError("embedding model crashed")
        seen.extend(docs)
        return orig(model, docs, **kwargs)
    [monkeypatch.setattr(cls, n, embed) for n in ("embed", "passage_embed")]
    return lambda: seen.extend([""] * nth)  # past nth: never raises again

def failed_forced_rebuild(world, harness, monkeypatch):  # B1/B2 setup: 105 indexed, forced rebuild dies at text 103 of 105
    client, gh = world(MANY)
    assert (ingest(client).status_code, len(points(harness))) == (200, 105)
    disarm = model_raises_at(monkeypatch, 103)
    ingest(client, force=True)
    return client, gh, disarm

def ask(client, repo=NAME):
    return lambda: client.post("/api/v1/ask", json={"question": "What does master_app return?", "repo_name": repo})

def test_ask_on_a_repo_during_its_full_reindex_gets_503(world):
    # Break: /ask never checks _busy_collections (query.py:43-50), so a mid-rebuild repo is read as if current.
    client, gh = world(FILES_A)
    assert ingest(client).status_code == 200
    early, _ = while_held(client, gh, ask(client), force=True)
    assert [(r.status_code, "being re-indexed, try again" in str(r.json().get("detail"))) for r in early] == [(503, True)]

def test_embedding_failure_after_the_delete_leaves_marker_0_so_the_next_ingest_rebuilds(world, harness, monkeypatch):
    # Break (gap I8): the failed build keeps the old compatible metadata, so the same-commit ingest takes the fast path.
    client, gh, disarm = failed_forced_rebuild(world, harness, monkeypatch)
    after_fail, before = stored_version(harness), len(gh.archives)
    disarm()
    assert ingest(client).status_code == 200
    assert (after_fail, len(gh.archives) - before, len(points(harness))) == (0, 1, 105)

def test_check_after_that_failure_reports_the_partial_index_as_not_compatible(world, harness, monkeypatch):
    # Break: /check trusts the old metadata (check.py:91), so 100 of 105 points read as a current index.
    client, _, _ = failed_forced_rebuild(world, harness, monkeypatch)
    body = client.get("/api/v1/check", params={"repo_name": NAME}).json()
    assert (body["indexed"], body["chunks"], body["compatible"]) == (True, 100, False)

def qdrant_raises(harness, monkeypatch, method, when):  # Qdrant wrapper: `method` raises on the call `when` picks
    real = getattr(harness.store, method)
    def call(*args, **kwargs):
        if when(*args, **kwargs):
            raise ConnectionError(f"qdrant refused {method}")
        return real(*args, **kwargs)
    monkeypatch.setattr(harness.store, method, call)

def test_failed_collection_delete_stops_the_rebuild_and_keeps_the_old_points(world, harness, monkeypatch):
    # Break: _delete_if_exists swallows the error (embedder.py:163-164), so d.py's point mixes into the old collection.
    client, gh = world(FIVE)
    assert (ingest(client).status_code, points(harness)) == (200, OLD_FIVE)
    gh.zips[None] = zip_of("demo-main", {**FIVE, "d.py": fn("d1")})
    qdrant_raises(harness, monkeypatch, "delete_collection", lambda *a, **k: NAME in (*a, k.get("collection_name")))
    status = ingest(client, force=True).status_code
    assert (status, points(harness), stored_version(harness)) == (500, OLD_FIVE, 0)

# P10-M1, P10-B4: brief puts them in tests/test_index_version.py (sealed by qf-p9-test); P6/P11 lanes update `2` here.
def test_forced_rebuild_that_succeeds_stores_index_version_2(world, harness):
    # Break (P10-M1): the full path passes 0, not INDEX_VERSION (ingest.py:597), so the marker is never replaced.
    client, gh = world(FILES_A)
    codes = (ingest(client).status_code, ingest(client, force=True).status_code)
    assert (codes, len(gh.archives), stored_version(harness)) == ((200, 200), 2, 2)

def test_failed_marker_write_stops_the_rebuild_before_the_delete(world, harness, monkeypatch):
    # Break (P10-B4): the delete runs before the version-0 marker, so a refused marker write leaves no old points.
    client, _ = world(FIVE)
    assert (ingest(client).status_code, points(harness)) == (200, OLD_FIVE)
    marker = lambda *a, **k: any((p.payload or {}).get("index_version") == 0 for p in k.get("points") or [])  # noqa: E731
    qdrant_raises(harness, monkeypatch, "upsert", marker)
    status = ingest(client, force=True).status_code
    assert (status, points(harness), stored_version(harness)) == (500, OLD_FIVE, 2)

def test_stored_points_keep_the_old_readers_shape(harness):
    # Break (P10-E2): a payload key or the dense vector name the 5e48365 reader uses is renamed or dropped.
    from core.embeddings.embedder import embed_and_store
    assert embed_and_store(harness.chunks, "shape")["chunks_stored"] == 2
    found, _ = harness.store.scroll("shape", limit=10, with_vectors=True)
    keys = ("text", "file_path", "chunk_type", "name", "start_line", "end_line", "language", "repo_name")
    assert sorted((list(p.vector), tuple(p.payload.get(k, "MISSING") for k in keys)) for p in found) == [
        (["fast-bge-small-en-v1.5"], ("def alpha(): return 0", "src/alpha.py", "function", "alpha", 1, 1, "python", "shape")),
        (["fast-bge-small-en-v1.5"], ("def beta(): return 1", "src/beta.py", "function", "beta", 1, 1, "python", "shape"))]

def test_ask_on_a_repo_during_its_incremental_update_answers_from_the_old_index(world):
    # Break (ruling 1): 503 for every busy ingest (_busy_collections holds every kind, ingest.py:330-341), not only a full re-index.
    client, gh = world(FILES_A)
    assert ingest(client).status_code == 200
    gh.sha, gh.zips[None] = "e" * 40, zip_of("demo-main", {**FILES_A, "util.py": FILES_A["util.py"].replace("9", "7")})
    early, out = while_held(client, gh, ask(client))
    assert ([(r.status_code, r.json().get("answer"), r.json().get("chunks_used")) for r in early], out[0].json()["incremental"]) == (
        [(200, "fake reply", 2)], True)
