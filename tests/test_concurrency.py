"""tests/test_concurrency.py — B6b non-blocking routes and ingest lock (seam B): one shared TestClient, threads, held download."""
import pytest
import requests
from fastapi.testclient import TestClient
from test_temp_folder import FILES_A, FakeGitHub, bg, zip_of

@pytest.fixture
def server(harness, monkeypatch, tmp_path):  # one client for every call (startup runs once); "other" indexed with 2 chunks
    harness.reload(env={"TMP_DIR": str(tmp_path)})
    monkeypatch.setattr("core.ingestion.cloner.TMP_DIR", str(tmp_path))
    monkeypatch.setattr(requests, "get", (gh := FakeGitHub({None: zip_of("demo-main", FILES_A)})).get)
    __import__("core.embeddings.embedder", fromlist=["x"]).embed_and_store(harness.chunks, "other")
    from main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, gh

def ingest(client, repo="demo", **body):
    return client.post("/api/v1/ingest", json={"github_url": f"https://github.com/o/{repo}", **body})
def while_held(client, gh, call, **body):  # demo's ingest waits at its download; `call` runs on another thread meanwhile
    gh.hold("demo")
    slow, out = bg(ingest, client, **body)
    try:
        assert gh.at["demo"].wait(5), "ingest never reached its download"
        t, box = bg(call)
        early = t.join(1.0) or list(box)  # join returns at once when the route answers; a frozen server waits the full 1 s
    finally:
        gh.gates["demo"].set(), slow.join(10)
    return t.join(10) or early, out

@pytest.mark.parametrize("repo, indexed", [("other", True), ("demo", False)])
def test_check_answers_while_an_ingest_waits_on_download(server, repo, indexed):
    # Break (D50): async def routes run blocking I/O on the one event loop (ingest.py:196), so /check waits for the ingest.
    client, gh = server
    early, _ = while_held(client, gh, lambda: client.get("/api/v1/check", params={"repo_name": repo}))
    assert [(r.status_code, r.json()["indexed"]) for r in early] == [(200, indexed)]

@pytest.mark.parametrize("question", ["What does alpha return?", "Where is beta defined?"])
def test_ask_on_another_indexed_repo_answers_while_an_ingest_runs(server, question):
    # Break (I2): /ask is async def (query.py:30) on the loop the held ingest blocks, so Q&A on "other" freezes too.
    client, gh = server
    early, _ = while_held(client, gh, lambda: client.post("/api/v1/ask", json={"question": question, "repo_name": "other"}))
    assert [(r.status_code, r.json()["answer"], r.json()["chunks_used"]) for r in early] == [(200, "fake reply", 2)]

@pytest.mark.parametrize("second", [{"repo": "tool"}, {"repo": "tool", "branch": "develop"}])
def test_ingests_of_two_different_repos_run_together(server, second):
    # Break: one global lock, or the blocked event loop (ingest.py:196), keeps tool's ingest from starting until demo's ends.
    client, gh = server
    tool = gh.hold("tool") or []
    early, demo = while_held(client, gh, lambda: tool.append(bg(ingest, client, **second)) or gh.at["tool"].wait(0.9))
    gh.gates["tool"].set(), tool[0][0].join(10)
    assert (early, [(r.status_code, r.json()["files_indexed"]) for r in demo + tool[0][1]]) == ([True], [(200, 2)] * 2)

@pytest.mark.parametrize("body", [{}, {"branch": "develop"}])
def test_second_ingest_of_same_collection_gets_409_at_once(server, body):
    # Break: no per-collection lock, so a second run of one collection clashes with the first (or, today, just waits).
    client, gh = server
    early, first = while_held(client, gh, lambda: (ingest(client, **body).status_code, len(gh.archives)), **body)
    assert (early, first[0].status_code) == ([(409, 1)], 200)  # refused while held, before any second download

@pytest.mark.parametrize("shas, want", [(["a" * 40, "b" * 40], ([200] * 2, 2)), (["a" * 40] * 3, ([200] * 3, 1))],
                         ids=["full-path-then-new-commit", "up-to-date-path-then-again"])
def test_finished_ingest_frees_the_lock_on_every_return_path(server, shas, want):
    # Guard (green today): bites if the lock is freed only on error, not on the early return (ingest.py:257-274).
    client, gh = server
    assert ([setattr(gh, "sha", s) or ingest(client).status_code for s in shas], len(gh.archives)) == want  # count = path

@pytest.mark.parametrize("fail, code", [("clone", 400), ("qdrant", 500)])
def test_ingest_that_raises_frees_the_lock(server, harness, monkeypatch, fail, code):
    # Guard (green today): bites if HTTPException (clone 400, ingest.py:286-290) or a bare Qdrant error leaks the lock.
    client, gh = server
    gh.status, real, fired = 404 if fail == "clone" else 200, harness.store.upsert, []
    def upsert(*args, **kwargs):  # Qdrant refuses the first metadata write only
        if fail == "qdrant" and kwargs.get("collection_name") == "repo_metadata" and not fired:
            raise fired.append(1) or ConnectionError("qdrant unreachable")
        return real(*args, **kwargs)
    monkeypatch.setattr(harness.store, "upsert", upsert)
    first, gh.status = ingest(client).status_code, 200
    assert (first, ingest(client).status_code) == (code, 200)
