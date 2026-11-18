"""tests/test_architecture_ok.py — B3b (AD13), seam C: ingest twice on the one harness store; fakes from test_llm_failures."""
from fastapi.testclient import TestClient
from qdrant_client import qdrant_fastembed
from test_llm_failures import FILES, GOOD_ARCH, NOT_FOUND, SHA1, SHA2, api, github, groq, ingest, stored  # noqa: F401

def embeds(monkeypatch):  # wraps the TB1 fake model's embed (also bound as passage_embed, which add() calls); logs each call
    fake, calls = qdrant_fastembed.TextEmbedding, []
    inner = fake.embed
    def counted(self, documents, **kwargs):
        calls.append(1)
        return inner(self, documents, **kwargs)
    for attr in ("embed", "passage_embed"):
        monkeypatch.setattr(fake, attr, counted)
    return calls

def test_failed_architecture_still_saves_the_graph_and_survives_a_restart(api, harness, monkeypatch):
    # middle, new (G6). Break: ingest.py:159-172 skips the whole save on failure -> nothing stored, /graph 404 after restart.
    groq(monkeypatch, NOT_FOUND)
    github(monkeypatch, SHA1)
    name = ingest(api).json()["repo_name"]
    record = stored(harness, name)
    from api.routes.ingest import graph_cache
    from main import app
    graph_cache.clear()  # restart stand-in: the module cache would otherwise answer /graph from memory
    with TestClient(app) as fresh:
        graph = fresh.get("/api/v1/graph", params={"repo_name": name})
    assert (record.get("architecture_ok"), record.get("commit_sha"), sorted(record.get("file_hashes", {})),
            len(record.get("graph_data") or {}) > 0, graph.status_code) == (False, SHA1, ["app.py", "util.py"], True, 200)

def test_next_ingest_at_same_sha_reruns_only_the_architecture(api, harness, monkeypatch):
    # middle, new. Break: retry takes the full path (re-embeds) or the fast path (no Groq call, flag stays false).
    server, calls = groq(monkeypatch, NOT_FOUND, GOOD_ARCH), embeds(monkeypatch)
    github(monkeypatch, SHA1)
    name = ingest(api).json()["repo_name"]
    first = len(calls)  # guard: the counter sees the first ingest's embedding, so a 0 below is real
    second = ingest(api)
    record = stored(harness, name)
    assert (first > 0, second.status_code, len(calls) - first, len(server.requests), record.get("architecture_ok"),
            record.get("mermaid")) == (True, 200, 0, 2, True, "flowchart TD\nA-->B")

def test_old_metadata_without_the_flag_counts_as_ok_and_keeps_the_fast_path(api, harness, monkeypatch):
    # edge, existing. Break: ingest.py:257 condition `False`, or a missing flag read as false -> re-download.
    from qdrant_client.models import Distance, PointStruct, VectorParams
    # the stored "demo" index under the current model's vector name (config.py:114), so the compatibility check passes
    harness.store.create_collection("demo", vectors_config={"fast-bge-small-en-v1.5": VectorParams(size=384, distance=Distance.COSINE)})
    harness.store.upsert("repo_metadata", [PointStruct(id=1, vector=[0.0], payload={  # a record saved before B3b
        "repo_name": "demo", "summary": "Old summary.", "mermaid": "flowchart TD\nX-->Y", "graph_stats": {},
        "graph_data": {}, "architecture_context": None, "commit_sha": SHA1, "file_hashes": {"app.py": "h1"},
        "index_version": 2})])
    server, urls = groq(monkeypatch), github(monkeypatch, SHA1)
    body = ingest(api).json()
    assert (sum("/archive/" in u for u in urls), len(server.requests), "already up to date" in body["message"]) == (0, 0, True)

def test_retry_that_fails_again_keeps_the_flag_false(api, harness, monkeypatch):
    # edge, new. Break: the retry writes architecture_ok true (or drops the key) whatever the reply, or saves the error text.
    server = groq(monkeypatch, NOT_FOUND, ("", "length"))
    github(monkeypatch, SHA1)
    name = ingest(api).json()["repo_name"]
    ingest(api)
    record = stored(harness, name)  # raw payload, so a reader-side default cannot hide a missing key
    assert (len(server.requests), "architecture_ok" in record, record.get("architecture_ok"), record.get("commit_sha"),
            record.get("summary") or "", record.get("mermaid") or "") == (2, True, False, SHA1, "", "")

def test_failed_flag_with_a_new_sha_goes_incremental_then_regenerates(api, harness, monkeypatch):
    # bad, new. Break: false flag read as an architecture-only retry (new files never indexed) or as no metadata (full).
    groq(monkeypatch, NOT_FOUND, GOOD_ARCH)
    github(monkeypatch, SHA1)
    name = ingest(api).json()["repo_name"]
    github(monkeypatch, SHA2, {**FILES, "util.py": "def helper():\n    return 2\n"})
    body = ingest(api).json()
    record = stored(harness, name)
    assert ((body["incremental"], body["files_modified"], body["files_added"]), record.get("commit_sha"),
            record.get("architecture_ok"), record.get("mermaid")) == ((True, 1, 0), SHA2, True, "flowchart TD\nA-->B")
