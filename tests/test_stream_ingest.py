"""tests/test_stream_ingest.py — P8 streamed ingest (seam A: POST /api/v1/ingest; fake GitHub zip, fake model, fake Groq, Qdrant :memory:)."""
import io, zipfile  # noqa: E401
from types import SimpleNamespace
import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from conftest import FakeTextEmbedding

SHA1, SHA2 = "1" * 40, "2" * 40
NAME = "o--demo"  # AD8 collection name for https://github.com/o/demo
GOOD = "SUMMARY: demo.\n```mermaid\nflowchart TD\nA-->B\n```"
BAD = "no diagram in this reply"  # no mermaid block -> architecture failed (architecture.py:105-107)

def fn(stem, i):  # one chunk: >= 50 chars (MIN_CHUNK_CHARS, chunker.py:126) [code]
    return f"def {stem}_{i}():\n    # a function long enough to pass MIN_CHUNK_CHARS\n    return {i}\n"

def repo(stems, n):  # {"<stem>.py": n functions}
    return {f"{s}.py": "".join(fn(s, i) for i in range(n)) for s in stems}

class Streamed:  # requests.Response as stream=True returns it (copied from test_download_stream.py)
    def __init__(self, status_code, body):
        self.status_code, self.headers, self.body = status_code, {"Content-Type": "application/zip"}, body
    def iter_content(self, chunk_size=1, decode_unicode=False):
        step = chunk_size or len(self.body) or 1
        yield from (self.body[i:i + step] for i in range(0, len(self.body), step))
    content = property(lambda self: self.body)
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")
    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass

class Recording(FakeTextEmbedding):  # fake model: records each list of documents it is asked to embed
    batches: list = []
    def passage_embed(self, documents, **kwargs):
        docs = list(documents)
        Recording.batches.append(docs)
        return (np.full(self.dim, 0.1) for _ in docs)
    embed = passage_embed

@pytest.fixture
def api(harness, monkeypatch, tmp_path):  # one app on the harness store; api.push(sha, files) sets what GitHub serves
    state = SimpleNamespace(sha=SHA1, zip=b"", replies=[])
    def push(sha, files):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for path, text in files.items():
                zf.writestr(f"demo-main/{path}", text)
        state.sha, state.zip = sha, buf.getvalue()
    def get(url, params=None, headers=None, timeout=None, allow_redirects=True, stream=False):
        if url.startswith("https://api.github.com/"):
            return SimpleNamespace(status_code=200, json=lambda: [{"sha": state.sha}], content=b"")
        return Streamed(200 if "/archive/" in url else 404, state.zip)
    real_create = harness.groq._create
    def create(**kwargs):  # fake Groq: next queued reply, GOOD once the queue is empty
        reply = real_create(**kwargs)
        reply.choices[0].message.content = state.replies.pop(0) if state.replies else GOOD
        return reply
    monkeypatch.setattr(harness.groq.chat.completions, "create", create)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(Recording, "batches", [])
    harness.model = Recording
    harness.reload(env={"TMP_DIR": str(tmp_path)})
    monkeypatch.setattr("core.ingestion.cloner.TMP_DIR", str(tmp_path), raising=False)
    from main import app
    client = TestClient(app)
    state.push = push
    state.post = lambda force=False: client.post("/api/v1/ingest", json={"github_url": "https://github.com/o/demo", "force": force})
    return state

def stored(harness):  # raw metadata payload for o--demo, {} when none
    if not harness.store.collection_exists("repo_metadata"):
        return {}
    points, _ = harness.store.scroll("repo_metadata", limit=10)
    return next((p.payload for p in points if p.payload["repo_name"] == NAME), {})

def names_in_store(harness):
    points, _ = harness.store.scroll(NAME, limit=2000)
    return {p.payload["name"] for p in points}

def arch_prompt(harness, i=-1):  # user message of the i-th request at fake Groq
    return harness.groq.requests[i]["messages"][-1]["content"]

def test_full_ingest_stores_every_chunk(api, harness):
    # P8-M1. Break: stream stops after the first upload group (100) or never flushes the last partial group (200).
    api.push(SHA1, repo("abcde", 50))  # 5 files x 50 functions = 250 chunks [measured, first qf-p8-test run]
    response = api.post()
    assert (response.status_code, response.json()["chunks_stored"], harness.store.count(NAME).count) == (200, 250, 250)
    assert {"e_0", "e_49"} <= names_in_store(harness)  # last file, first and last function

def incremental(api, harness):  # 3 files x 2 functions indexed at SHA1; then b.py changes to 2 new functions at SHA2
    api.push(SHA1, repo("abc", 2))
    assert (api.post().status_code, harness.store.count(NAME).count) == (200, 6)
    api.push(SHA2, {**repo("abc", 2), "b.py": fn("bnew", 0) + fn("bnew", 1)})

B_NEW = ["def bnew_0():\n    # a function long enough to pass MIN_CHUNK_CHARS\n    return 0",
         "def bnew_1():\n    # a function long enough to pass MIN_CHUNK_CHARS\n    return 1"]  # chunk text = source minus last newline [measured]

def test_incremental_ingest_embeds_only_the_changed_file(api, harness):
    # P8-M2. Break: unchanged a.py and c.py are embedded again (ingest.py:485 -> chunk_files(file_list)).
    incremental(api, harness)
    Recording.batches.clear()
    response = api.post()
    assert (response.status_code, sorted(d for b in Recording.batches for d in b)) == (200, B_NEW)

FIFTEEN = [f"m{i:02}" for i in range(1, 16)]  # m01.py .. m15.py; no name is a substring of another

def missing_files(prompt):
    return [f"{s}.py" for s in FIFTEEN if f"{s}.py" not in prompt]

def test_full_ingest_architecture_input_lists_every_walked_file(api, harness):
    # P8-M3. Break: input built from a 12-file sample of chunks (context_selector.py:64-89, max_chunks=12).
    api.push(SHA1, repo(FIFTEEN, 2))
    response = api.post()
    assert (response.status_code, len(harness.groq.requests), missing_files(arch_prompt(harness))) == (200, 1, [])

def test_incremental_architecture_names_every_file_with_code_only_from_changed(api, harness):
    # P8-M4 (ruling 2). Break: unchanged a.py, c.py re-chunked for the architecture (ingest.py:513), or no code sent at all.
    incremental(api, harness)
    response = api.post()
    prompt = arch_prompt(harness)
    assert (response.status_code, len(harness.groq.requests)) == (200, 2)
    assert ([f for f in ("a.py", "b.py", "c.py") if f in prompt], B_NEW[0] in prompt) == (["a.py", "b.py", "c.py"], True)
    assert [body for body in ("def a_0():", "def a_1():", "def c_0():", "def c_1():") if body in prompt] == []

def test_1001_chunks_are_stored_not_refused(api, harness):
    # P8-E1. Break: the 1,000-chunk refusal (embedder.py:177) still answers 500 "Repository too large".
    api.push(SHA1, repo("abcdefg", 143))  # 7 files x 143 functions = 1,001 chunks [code: one chunk per function]
    response = api.post()
    assert (response.status_code, response.json().get("chunks_stored"), harness.store.collection_exists(NAME)
            and harness.store.count(NAME).count) == (200, 1001, 1001)

def test_architecture_retry_lists_every_file(api, harness):
    # P8-E4. Break: the retry path samples scrolled chunks (ingest.py:221, 12 files; gap I27).
    api.push(SHA1, repo(FIFTEEN, 2))
    api.replies.append(BAD)  # first architecture fails; the same-SHA ingest then retries only the architecture
    first, retry = api.post(), api.post()
    assert (first.status_code, retry.status_code, len(harness.groq.requests)) == (200, 200, 2)
    assert missing_files(arch_prompt(harness, 1)) == []

def test_forced_reingest_of_stub_files_fails_and_keeps_the_old_index(api, harness):
    # P8-E5. Break: an empty chunk stream is never checked (ingest.py:547 -> if False), so the old index is touched first.
    api.push(SHA1, repo("abc", 2))
    assert (api.post().status_code, harness.store.count(NAME).count) == (200, 6)
    api.push(SHA2, {"a.py": "x = 1\n", "b.py": "y = 2\n", "c.py": "z = 3\n"})  # each under 50 chars -> 0 chunks [measured]
    response = api.post(force=True)
    assert (response.status_code, "Chunking produced no output" in response.text, harness.store.count(NAME).count,
            stored(harness).get("commit_sha")) == (500, True, 6, SHA1)

def test_failed_stale_delete_stops_the_incremental_run(api, harness, monkeypatch):
    # P8-B1. Break: delete failure printed and ignored (embedder.py:311-313); run saves SHA2 and new hashes.
    incremental(api, harness)
    before = stored(harness)["file_hashes"]
    def refuse(*args, **kwargs):  # Qdrant point delete raises
        raise RuntimeError("qdrant delete refused")
    monkeypatch.setattr(harness.store, "delete", refuse)
    response = api.post()
    after = stored(harness)
    assert (response.status_code, after.get("commit_sha"), after.get("file_hashes") == before) == (500, SHA1, True)

def test_embedding_never_takes_more_than_one_text_at_a_time(api, harness):
    # P8-B2. Break: a whole upload group (100) or file list sent to the model at once (3.3 GB at batch 100 [measured, AD7]).
    api.push(SHA1, repo("abcde", 50))  # 250 chunks, default settings
    response = api.post()
    sizes = [len(b) for b in Recording.batches]
    assert (response.status_code, max(sizes, default=0), sum(sizes)) == (200, 1, 250)

def test_qdrant_refusing_a_later_upload_group_fails_the_first_ingest(api, harness, monkeypatch):
    # P8-B3. Break: the upload error is swallowed (ingest.py:562 -> if False) and metadata saved as if complete.
    real_upsert, groups = harness.store.upsert, []
    def upsert(collection_name, points, **kwargs):  # Qdrant refuses the 2nd group of chunk points
        if collection_name == NAME:
            groups.append(len(points))
            if len(groups) == 2:
                raise RuntimeError("qdrant refused group 2")
        return real_upsert(collection_name, points, **kwargs)
    monkeypatch.setattr(harness.store, "upsert", upsert)
    api.push(SHA1, repo("abcde", 50))  # 250 chunks -> 3 upload groups of <= 100 [code: UPLOAD_GROUP]
    response = api.post()
    assert (response.status_code, "qdrant refused group 2" in response.text, stored(harness)) == (500, True, {})
