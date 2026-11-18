"""tests/test_cloner_download.py — B7 default-branch download (seam A); requests.get faked at the network boundary."""
import io, zipfile  # noqa: E401
from types import SimpleNamespace
import pytest
import requests
from fastapi.testclient import TestClient
from qdrant_client.models import Distance, VectorParams
from test_download_stream import Streamed  # archive answers in the stream=True form (P9)

SHA = "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d"
APP = {f"{n}.py": f"def {n}(items):\n    total = sum(len(str(i)) for i in items)\n    return total\n" for n in ("app", "util")}

def zip_of(top, files):  # GitHub archive layout: every file under one top folder
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(f"{top}/{name}", text)
    return buf.getvalue()

class FakeGitHub:  # api.github.com plus github.com archives at requests.get; records every URL asked, in order
    def __init__(self, served, sha=SHA, api_status=200):
        self.served, self.sha, self.api_status, self.urls = served, sha, api_status, []
    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True, stream=False):
        self.urls.append(url)
        if url.startswith("https://api.github.com/"):
            body = ([{"sha": self.sha}] if self.sha else []) if self.api_status == 200 else {"message": "API rate limit exceeded"}
            return SimpleNamespace(status_code=self.api_status, json=lambda: body, content=b"")
        hit = self.served.get(url)
        return Streamed(200 if hit else 404, [hit or b"Not Found"])
    @property
    def archives(self):
        return [u.rsplit("/archive/", 1)[1] for u in self.urls if "/archive/" in u]

def use_fake(harness, monkeypatch, tmp_path, gh):
    # A Qdrant that already served an ingest holds `repo_metadata`; keeps the B1b empty-Qdrant crash out of these reds.
    harness.store.create_collection("repo_metadata", vectors_config=VectorParams(size=1, distance=Distance.COSINE))
    monkeypatch.setattr(requests, "get", gh.get)
    harness.reload(env={"TMP_DIR": str(tmp_path)})
    monkeypatch.setattr("core.ingestion.cloner.TMP_DIR", str(tmp_path), raising=False)

def ingest(harness, monkeypatch, tmp_path, gh, url):
    use_fake(harness, monkeypatch, tmp_path, gh)
    from main import app
    return TestClient(app).post("/api/v1/ingest", json={"github_url": url})

@pytest.mark.parametrize("owner, repo, sha", [
    pytest.param("o", "demo", SHA, id="develop-default"),
    pytest.param("acme", "tool", "0123456789abcdef0123456789abcdef01234567", id="trunk-default"),
])
def test_default_branch_not_main_downloads_resolved_sha_archive(harness, monkeypatch, tmp_path, owner, repo, sha):
    # Break: default branch download ignores the resolved SHA and asks only refs/heads/main + master (today).
    base = f"https://github.com/{owner}/{repo}/archive"
    gh = FakeGitHub({f"{base}/{sha}.zip": zip_of(f"{repo}-{sha}", APP)}, sha=sha)
    resp = ingest(harness, monkeypatch, tmp_path, gh, f"https://github.com/{owner}/{repo}")
    assert (resp.status_code, resp.json().get("files_indexed"), gh.archives) == (200, 2, [f"{sha}.zip"])

@pytest.mark.parametrize("owner, repo", [pytest.param("o", "demo", id="o-demo"), pytest.param("pallets", "flask", id="pallets-flask")])
def test_default_branch_main_still_downloads(harness, monkeypatch, tmp_path, owner, repo):
    # Break: main archive path wrong (cloner.py:291) -> every default-branch repo fails; guard, green today.
    base, z = f"https://github.com/{owner}/{repo}/archive", zip_of(f"{repo}-main", APP)
    gh = FakeGitHub({f"{base}/{SHA}.zip": z, f"{base}/HEAD.zip": z, f"{base}/refs/heads/main.zip": z})
    resp = ingest(harness, monkeypatch, tmp_path, gh, f"https://github.com/{owner}/{repo}")
    assert (resp.status_code, resp.json().get("files_indexed"), resp.json().get("chunks_stored", 0) > 0) == (200, 2, True)

@pytest.mark.parametrize("api_status, sha", [pytest.param(403, SHA, id="rate-limited-403"), pytest.param(200, None, id="empty-commit-list")])
def test_no_sha_tries_head_zip_before_main_and_master(harness, monkeypatch, tmp_path, api_status, sha):
    # Break: with no resolved SHA the download skips HEAD.zip and guesses refs/heads/main first (today).
    base, z = "https://github.com/o/demo/archive", zip_of("demo-develop", APP)
    gh = FakeGitHub({f"{base}/HEAD.zip": z, f"{base}/refs/heads/main.zip": z, f"{base}/refs/heads/master.zip": z},
                    sha=sha, api_status=api_status)
    resp = ingest(harness, monkeypatch, tmp_path, gh, "https://github.com/o/demo")
    assert (resp.status_code, resp.json().get("files_indexed"), gh.archives) == (200, 2, ["HEAD.zip"])

@pytest.mark.parametrize("served, want", [
    pytest.param("refs/heads/main.zip", [f"{SHA}.zip", "HEAD.zip", "refs/heads/main.zip"], id="sha-and-head-404-then-main"),
    pytest.param("refs/heads/master.zip", [f"{SHA}.zip", "HEAD.zip", "refs/heads/main.zip", "refs/heads/master.zip"],
                 id="all-but-master-404"),
])
def test_failed_sha_archive_falls_back_in_order(harness, monkeypatch, tmp_path, served, want):
    # Break: SHA passed only as commit_hash (one URL, cloner.py:280-282), or HEAD.zip/main/master skipped or reordered.
    gh = FakeGitHub({f"https://github.com/o/demo/archive/{served}": zip_of("demo-main", APP)})
    resp = ingest(harness, monkeypatch, tmp_path, gh, "https://github.com/o/demo")
    assert (resp.status_code, resp.json().get("files_indexed"), gh.archives) == (200, 2, want)

@pytest.mark.parametrize("repo", ["legacy", "old-tool"])
def test_master_only_repo_without_sha_still_downloads(harness, monkeypatch, tmp_path, repo):
    # Break: master dropped from the fallback list (cloner.py:292) -> old master-only repos fail; guard, green today.
    gh = FakeGitHub({f"https://github.com/o/{repo}/archive/refs/heads/master.zip": zip_of(f"{repo}-master", APP)},
                    api_status=403)
    resp = ingest(harness, monkeypatch, tmp_path, gh, f"https://github.com/o/{repo}")
    assert (resp.status_code, resp.json().get("files_indexed"), gh.archives[-1]) == (200, 2, "refs/heads/master.zip")

@pytest.mark.parametrize("code", [404, 403])
def test_every_archive_missing_reports_could_not_download_with_status(harness, monkeypatch, tmp_path, code):
    # Break: error text lost or status not reported (cloner.py:310-311) -> UI cannot show why (AC8); guard, green today.
    use_fake(harness, monkeypatch, tmp_path, SimpleNamespace(get=lambda url, **kw: Streamed(code, [b""])))
    from core.ingestion.cloner import clone_repo
    msg = clone_repo("https://github.com/o/gone", "o", "gone")["message"]
    assert (msg.startswith("Could not download repo 'https://github.com/o/gone'"), msg.endswith(f"HTTP {code}.")) == (True, True), msg

@pytest.mark.parametrize("exc", [requests.ConnectTimeout, requests.ReadTimeout])
def test_download_timeout_reports_timeout_message(harness, monkeypatch, tmp_path, exc):
    # Break: Timeout re-raised or swallowed as "Unexpected error" (cloner.py:349) -> 500 or vague text; guard, green today.
    def slow(url, **kw):
        raise exc(f"HTTPSConnectionPool(host='github.com'): timed out ({url})")
    use_fake(harness, monkeypatch, tmp_path, SimpleNamespace(get=slow))
    from core.ingestion.cloner import clone_repo
    got = clone_repo("https://github.com/o/huge", "o", "huge")
    assert (got["status"], got["message"]) == ("error", "Download timed out after 60s. Repository may be too large.")
