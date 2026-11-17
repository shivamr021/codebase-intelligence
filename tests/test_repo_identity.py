"""tests/test_repo_identity.py — B8 owner-aware repo identity (seam B); GitHub faked at requests.get."""
import contextlib, threading  # noqa: E401
from concurrent.futures import ThreadPoolExecutor
import pytest
import requests
from fastapi.testclient import TestClient
from test_cloner_download import APP, SHA, FakeGitHub, use_fake, zip_of

def served(owner, repo, files, *refs):  # what GitHub serves for one repo: SHA, HEAD, main and any named branch
    z, base = zip_of(f"{repo}-main", files), f"https://github.com/{owner}/{repo}/archive"
    return {f"{base}/{r}.zip": z for r in (SHA, "HEAD", "refs/heads/main", *refs)}

def client(harness, monkeypatch, tmp_path, archives):
    use_fake(harness, monkeypatch, tmp_path, FakeGitHub(archives))
    from main import app
    return TestClient(app, raise_server_exceptions=False)

def post(c, **body):  # a plain-text 500 body becomes {"detail": text}, so the assert shows it
    r = c.post("/api/v1/ingest", json=body)
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text}

def name_of(monkeypatch, url):
    monkeypatch.setattr(requests, "get", FakeGitHub({}).get)
    from core.ingestion.cloner import resolve_repo
    return resolve_repo(url)["repo_name"]

@pytest.mark.parametrize("url, want", [
    ("https://github.com/tiangolo/fastapi", "tiangolo--fastapi"),
    ("https://github.com/pallets/flask", "pallets--flask"),
    ("https://github.com/shivamr021/codebase-intelligence", "shivamr021--codebase-intelligence"),
])
def test_repo_name_is_owner_and_repo(harness, monkeypatch, url, want):
    # Break: name drops the owner (cloner.py:241 "repo_name": repo, today) -> two owners share one collection.
    assert name_of(monkeypatch, url) == want

@pytest.mark.parametrize("owner, repo, want", [("tiangolo", "fastapi", "tiangolo--fastapi"), ("pallets", "flask", "pallets--flask")])
def test_check_by_github_url_finds_the_collection_ingest_made(harness, monkeypatch, tmp_path, owner, repo, want):
    # Break: /check still requires repo_name (check.py:40) or derives a name ingest did not use -> 422 / indexed False.
    c, url = client(harness, monkeypatch, tmp_path, served(owner, repo, APP)), f"https://github.com/{owner}/{repo}"
    made = post(c, github_url=url)
    got = c.get("/api/v1/check", params={"github_url": url})
    assert (made.get("repo_name"), got.status_code, got.json().get("indexed"), got.json().get("chunks")) == (
        want, 200, True, made.get("chunks_stored"))

@pytest.mark.parametrize("base", ["https://github.com/tiangolo/fastapi", "https://github.com/octocat/Hello-World"])
def test_plain_dot_git_and_trailing_slash_give_one_name(harness, monkeypatch, base):
    # Break: .git kept (cloner.py:162), or both "/" strips dropped (:161 + :164) -> one repo, two collections; guard.
    names = [name_of(monkeypatch, u) for u in (base, base + ".git", base + "/")]
    assert len(set(names)) == 1, names

@pytest.mark.parametrize("url, want", [
    ("https://github.com/octocat/octocat.github.io", "octocat--octocat.github.io"),
    ("https://github.com/vercel/next.js", "vercel--next.js"),
])
def test_dots_in_repo_name_are_kept(harness, monkeypatch, url, want):
    # Break: name sanitizer strips or swaps "." -> octocat.github.io and octocatgithubio collide; Qdrant allows ".".
    assert name_of(monkeypatch, url) == want

@pytest.mark.parametrize("forms", [
    ("https://github.com/tiangolo/fastapi", "https://github.com/Tiangolo/FastAPI", "https://github.com/TIANGOLO/fastapi"),
    ("https://github.com/octocat/hello-world", "https://github.com/octocat/Hello-World"),
])
def test_owner_and_repo_case_map_to_one_collection(harness, monkeypatch, forms):
    # Break: case kept in owner or repo (cloner.py:241 today) -> GitHub's one repo indexed twice, /check misses it.
    names = [name_of(monkeypatch, u) for u in forms]
    assert len(set(names)) == 1, names

def test_two_owners_same_repo_ingested_together_get_own_collections(harness, monkeypatch, tmp_path):
    # Break: owner dropped from the name (cloner.py:241) or shared TMP_DIR/<repo> folder (cloner.py:268-275, B6a).
    c = client(harness, monkeypatch, tmp_path, {**served("alice", "utils", {"a.py": APP["app.py"]}), **served("bob", "utils", APP)})
    gate, get = threading.Barrier(2, timeout=5), requests.get
    def both_downloading(url, **kw):  # each run's download waits until the other run is downloading too
        with contextlib.suppress(threading.BrokenBarrierError):
            "/archive/" in url and gate.wait()
        return get(url, **kw)
    monkeypatch.setattr(requests, "get", both_downloading)
    with ThreadPoolExecutor(2) as pool:
        out = list(pool.map(lambda o: post(c, github_url=f"https://github.com/{o}/utils"), ("alice", "bob")))
    got = [(r.get("repo_name"), r.get("files_indexed"), r.get("detail")) for r in out]
    assert got == [("alice--utils", 1, None), ("bob--utils", 2, None)]
    assert [harness.store.count(r["repo_name"]).count for r in out] == [r["chunks_stored"] for r in out]

@pytest.mark.parametrize("branch, twin", [("develop", "foo__develop"), ("feature/x", "foo__feature-x")])
def test_branch_suffix_cannot_collide_with_a_repo_name(harness, monkeypatch, tmp_path, branch, twin):
    # Break: branch joined as "__<branch>" onto the bare repo (ingest.py:104-110) -> o/foo@branch and o/<twin> share one.
    c = client(harness, monkeypatch, tmp_path, {**served("o", "foo", {"a.py": APP["app.py"]}, f"refs/heads/{branch}"), **served("o", twin, APP)})
    a, b = post(c, github_url="https://github.com/o/foo", branch=branch), post(c, github_url=f"https://github.com/o/{twin}")
    assert (a.get("repo_name") != b.get("repo_name"), a.get("files_indexed"), b.get("files_indexed")) == (True, 1, 2), (a, b)
    assert [harness.store.count(r["repo_name"]).count for r in (a, b)] == [a["chunks_stored"], b["chunks_stored"]]
