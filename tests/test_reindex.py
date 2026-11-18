"""tests/test_reindex.py — B4a: force and auto re-index (seam A); GitHub faked at requests.get, Groq at the harness fake."""
import hashlib, io, zipfile  # noqa: E401
from types import SimpleNamespace
import pytest
import requests
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from qdrant_client import models as qm

OLD, NEW, OLD_VEC = "a" * 40, "b" * 40, "fast-bge-small-en"
BODY = "def {}():\n    # a function long enough to pass MIN_CHUNK_CHARS (chunker.py:126)\n    return {}\n"
FILES = {"app.py": BODY.format("alpha", 1), "util.py": BODY.format("beta", 2)}
HASHES = {p: hashlib.sha256(t.encode()).hexdigest() for p, t in FILES.items()}  # what the walker stores for FILES
FRESH = {("app.py", OLD), ("util.py", OLD)}  # (file_path, commit_hash) of every point after a full index at OLD
CHANGED = {**FILES, "util.py": BODY.format("beta", 3)}  # NEW commit: util.py modified, app.py same
AT_NEW = {("app.py", NEW), ("util.py", NEW)}
CUR_VEC = {"fast-bge-small-en-v1.5"}  # vector name once B1b sets the model (PROOF-blueprint.md:63)

class FakeGitHub:  # api.github.com + github.com archives behind requests.get: records every URL
    def __init__(self, sha, archive=200, files=FILES):
        self.sha, self.archive, self.urls, buf = sha, archive, [], io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for path, text in files.items():
                zf.writestr(f"demo-main/{path}", text)
        self.zip = buf.getvalue()
    def get(self, url, **kwargs):
        self.urls.append(url)
        if "/archive/" in url:
            return SimpleNamespace(status_code=self.archive, content=self.zip if self.archive == 200 else b"Not Found")
        body = [{"sha": self.sha}] if url.endswith("/commits") else {"sha": self.sha, "default_branch": "main"}
        return SimpleNamespace(status_code=200, json=lambda: body)
    archives = property(lambda self: [u for u in self.urls if "/archive/" in u])

def arch_reply(**kwargs):  # a valid architecture answer, so the metadata save never depends on B3a/B3b
    msg = ChatCompletionMessage(role="assistant", content="Demo summary.\n```mermaid\nflowchart TD\nA-->B\n```")
    return ChatCompletion(id="c1", object="chat.completion", created=0, model=kwargs["model"],
                          choices=[Choice(index=0, finish_reason="stop", message=msg)])

def seed(h, sha=OLD, vector=None, version=2, hashes=HASHES):
    # Stored index for "demo": vector=None -> the app's own embed_and_store (current name); else a raw old collection.
    if vector:
        h.store.create_collection("demo", vectors_config={vector: qm.VectorParams(size=384, distance=qm.Distance.COSINE)})
        h.store.upsert("demo", [qm.PointStruct(id=i, vector={vector: [0.1] * 384}, payload={**c, "commit_hash": sha})
                                for i, c in enumerate(h.chunks)])
    else:
        from core.embeddings.embedder import embed_and_store
        embed_and_store(h.chunks, "demo", commit_hash=sha)
    from core.storage.repo_metadata import ensure_metadata_collection
    ensure_metadata_collection()
    payload = {"repo_name": "demo", "summary": "s", "mermaid": "flowchart TD\nA-->B", "graph_stats": {"nodes": 2},
               "graph_data": {"nodes": ["app.py", "util.py"], "edges": []}, "architecture_context": None,
               "commit_sha": sha, "file_hashes": hashes, **({} if version is None else {"index_version": version})}
    h.store.upsert("repo_metadata", [qm.PointStruct(id=hashlib.md5(b"demo").hexdigest(), vector=[0.0], payload=payload)])

def stored(h):  # (file_path, commit_hash) of every point in "demo", plus its vector names
    points = h.store.scroll("demo", limit=100, with_payload=True)[0]
    vectors = h.store.get_collection("demo").config.params.vectors
    return {(p.payload["file_path"], p.payload.get("commit_hash")) for p in points}, set(vectors)

@pytest.fixture
def ingest(harness, monkeypatch, tmp_path):
    monkeypatch.setattr(__import__("core.ingestion.cloner", fromlist=["x"]), "TMP_DIR", str(tmp_path))
    harness.groq.chat.completions.create = arch_reply
    from main import app
    def run(github, **body):
        monkeypatch.setattr(requests, "get", github.get)
        with TestClient(app) as client:
            return client.post("/api/v1/ingest", json={"github_url": "https://github.com/o/demo", **body})
    return run

def test_force_on_up_to_date_repo_runs_full_reindex(harness, ingest):
    # Break: `force` ignored -> fast path at ingest.py:257 returns "already up to date", no download, old points kept.
    seed(harness)
    r = ingest(gh := FakeGitHub(OLD), force=True).json()
    assert (r["incremental"], r["files_indexed"], len(gh.archives) >= 1) == (False, 2, True)
    assert stored(harness)[0] == FRESH

def test_up_to_date_repo_without_force_takes_fast_path(harness, ingest):
    # Break: ingest.py:257 never true -> download happens, lands at :335-361 "content unchanged", incremental True.
    seed(harness)
    r = ingest(gh := FakeGitHub(OLD)).json()
    assert (len(gh.archives), r["incremental"], "is already up to date" in r["message"]) == (0, False, True)

def test_old_vector_name_same_sha_runs_full_reindex(harness, ingest):
    # Break: fast path (ingest.py:257) trusts the SHA alone (G2) -> 0 files, old src/* points under the old name kept.
    seed(harness, vector=OLD_VEC)
    r = ingest(gh := FakeGitHub(OLD)).json()
    assert (r["incremental"], r["files_indexed"], len(gh.archives) >= 1) == (False, 2, True)
    assert stored(harness) == (FRESH, CUR_VEC)

def test_old_vector_name_new_sha_runs_full_reindex_not_incremental(harness, ingest):
    # Break: incremental path (ingest.py:314) add()s into the old collection (embedder.py:282) -> src/* points survive.
    seed(harness, vector=OLD_VEC)
    r = ingest(FakeGitHub(NEW, files=CHANGED)).json()
    assert (r["incremental"], r["files_indexed"], stored(harness)) == (False, 2, (AT_NEW, CUR_VEC))

@pytest.mark.parametrize("version, sha, points", [(1, OLD, FRESH), (None, NEW, AT_NEW)], ids=["version-1-same-sha", "version-missing-new-sha"])
def test_old_index_version_runs_full_reindex(harness, ingest, version, sha, points):
    # Break: index_version ignored (G3) -> fast path (same SHA) or incremental "content unchanged" (new SHA), src/* kept.
    seed(harness, version=version)
    r = ingest(FakeGitHub(sha)).json()
    assert (r["incremental"], r["files_indexed"], stored(harness)[0]) == (False, 2, points)

def test_force_on_never_indexed_repo_does_normal_full_index(harness, ingest):
    # Guard (passes today: force is an unknown field): a force path that drops a missing collection must not fail.
    r = ingest(FakeGitHub(OLD), force=True)
    assert (r.status_code, r.json().get("incremental"), r.json().get("files_indexed")) == (200, False, 2), r.text
    assert stored(harness)[0] == FRESH

def test_current_index_new_sha_takes_incremental_path(harness, ingest):
    # Break: ingest.py:314 `if False:` (or "any SHA change -> full") -> incremental False, no files_modified.
    seed(harness)
    r = ingest(FakeGitHub(NEW, files=CHANGED)).json()
    assert (r["incremental"], r["files_added"], r["files_modified"], r["files_deleted"]) == (True, 0, 1, 0)

def test_branch_plus_commit_hash_with_force_is_rejected(harness, ingest):
    # Break: models.py:53 check off -> request reaches GitHub (commit lookup) instead of a 422.
    r = ingest(gh := FakeGitHub(OLD), force=True, branch="main", commit_hash=OLD)
    assert (r.status_code, len(gh.urls)) == (422, 0)

def test_force_with_failed_download_returns_400_and_keeps_old_index(harness, ingest):
    # Break: force ignored -> fast path 200 (seeded SHA == served SHA); or force wipes index/metadata before clone fails.
    seed(harness)
    r = ingest(gh := FakeGitHub(OLD, archive=404), force=True)
    meta = harness.store.retrieve("repo_metadata", [hashlib.md5(b"demo").hexdigest()])[0].payload
    # archive order and count are B7's (tests/test_cloner_download.py:65-82); here only "a download was tried"
    assert (r.status_code, len(gh.archives) >= 1, meta["commit_sha"]) == (400, True, OLD)
    assert stored(harness)[0] == {("src/alpha.py", OLD), ("src/beta.py", OLD)}
