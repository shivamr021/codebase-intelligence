"""tests/test_index_version.py — B2b stored index_version (seam B: POST /api/v1/ingest, requests.get faked; seam C: read_index_version)."""
import hashlib, io, zipfile  # noqa: E401
from types import SimpleNamespace
import pytest
import requests
from fastapi.testclient import TestClient
from qdrant_client.models import Distance, PointStruct, VectorParams

SHA = "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d"
APP = "def total(items):\n    count = sum(len(str(item)) for item in items)\n    return count\n"
REPLY = "SUMMARY: one module app.py that totals item lengths.\n```mermaid\nflowchart TD\nA[app.py] --> B[total]\n```"

class FakeGitHub:  # api.github.com (HEAD sha) plus github.com archives at requests.get; records every URL asked
    def __init__(self):
        self.urls, buf = [], io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("demo-main/app.py", APP)
        self.zip = buf.getvalue()
    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
        self.urls.append(url)
        if url.startswith("https://api.github.com/"):
            return SimpleNamespace(status_code=200, json=lambda: [{"sha": SHA}], content=b"")
        return SimpleNamespace(status_code=200 if "/archive/" in url else 404, content=self.zip)

def ingest(harness, monkeypatch, tmp_path, seed=None):
    gh, real_create = FakeGitHub(), harness.groq._create
    def create(**kwargs):  # same fake Groq server, this test's architecture reply
        reply = real_create(**kwargs)
        reply.choices[0].message.content = REPLY
        return reply
    monkeypatch.setattr(harness.groq.chat.completions, "create", create)
    monkeypatch.setattr(requests, "get", gh.get)
    harness.reload(env={"TMP_DIR": str(tmp_path)})
    monkeypatch.setattr("core.ingestion.cloner.TMP_DIR", str(tmp_path), raising=False)
    if seed is not None:  # metadata as a previous ingest left it, written straight to the store
        harness.store.create_collection("repo_metadata", vectors_config=VectorParams(size=1, distance=Distance.COSINE))
        # the stored "demo" index, under the current model's vector name (config.py:114), so only the version can differ
        harness.store.create_collection("demo", vectors_config={"fast-bge-small-en-v1.5": VectorParams(size=384, distance=Distance.COSINE)})
        payload = {"repo_name": "demo", "summary": "old summary", "mermaid": "flowchart TD\nA-->B", "graph_stats": {},
                   "graph_data": {}, "architecture_context": None, **seed}
        harness.store.upsert("repo_metadata", [PointStruct(id=hashlib.md5(b"demo").hexdigest(), vector=[0.0], payload=payload)])
    from main import app
    return TestClient(app).post("/api/v1/ingest", json={"github_url": "https://github.com/o/demo"}), gh

def stored_version(harness):  # raw payload key, never through a reader that maps missing to 1
    points, _ = harness.store.scroll("repo_metadata", limit=10)
    assert [p.payload["repo_name"] for p in points] == ["demo"]
    return points[0].payload.get("index_version", "MISSING")

def test_full_ingest_stores_index_version_2(harness, monkeypatch, tmp_path):
    # Break: save_repo_metadata writes no index_version (today, ingest.py:160-169), so old and new chunks look alike.
    response, _ = ingest(harness, monkeypatch, tmp_path)
    assert (response.status_code, response.json()["incremental"]) == (200, False)
    assert stored_version(harness) == 2

def test_fast_path_same_sha_leaves_stored_version_unchanged_and_downloads_nothing(harness, monkeypatch, tmp_path):
    # Break: fast path skipped (`ingest.py:260` -> False), archive fetched and payload rewritten without index_version.
    response, gh = ingest(harness, monkeypatch, tmp_path, seed={"commit_sha": SHA, "file_hashes": {"app.py": "0" * 64}, "index_version": 2})
    assert (response.status_code, [u for u in gh.urls if "/archive/" in u]) == (200, [])
    assert stored_version(harness) == 2

@pytest.mark.parametrize("stored_hash", [
    pytest.param("0" * 64, id="file-modified"),  # ingest.py:404-407 rewrites via _run_graph_and_architecture
    pytest.param(hashlib.sha256(APP.encode()).hexdigest(), id="content-unchanged"),  # ingest.py:339-348
])
def test_incremental_ingest_leaves_stored_version_unchanged(harness, monkeypatch, tmp_path, stored_hash):
    # Break: the incremental save drops index_version (today), or overwrites the stored version it never rebuilt.
    seed = {"commit_sha": "0123456789abcdef0123456789abcdef01234567", "file_hashes": {"app.py": stored_hash}, "index_version": 2}
    response, _ = ingest(harness, monkeypatch, tmp_path, seed=seed)
    assert (response.status_code, response.json()["incremental"]) == (200, True)
    assert stored_version(harness) == 2

def read(metadata):
    from core.ingestion.chunker import read_index_version  # B2b's new name; imported here so one missing name reds only these
    return read_index_version(metadata)

@pytest.mark.parametrize("metadata, want", [
    ({"commit_sha": SHA, "file_hashes": {}}, 1), ({}, 1), (None, 1),
    ({"index_version": 2}, 2), ({"index_version": 3}, 3),  # controls: a hardcoded `return 1` fails here
])
def test_metadata_without_index_version_reads_as_version_1(metadata, want):
    # Break: reader returns None or raises KeyError on pre-B2b metadata, so old collections never count as stale.
    assert read(metadata) == want

@pytest.mark.parametrize("value", ["abc", None, [2]])
def test_hand_edited_non_int_index_version_reads_as_1_without_crashing(value):
    # Break: `int("abc")` raises ValueError, or a non-int is passed through as the version.
    assert (read({"index_version": value}), read({"index_version": 2})) == (1, 2)
