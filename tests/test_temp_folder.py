"""tests/test_temp_folder.py — B6a own temp folder per ingest (seam A): ingest_repo from threads, held or failed at embed."""
import asyncio, hashlib, inspect, io, os, threading, zipfile  # noqa: E401
from types import SimpleNamespace
import pytest
import requests
from fastapi import HTTPException
import fastembed  # the harness binds the fake model here for both qdrant-client 1.9.1 and 1.20.0
from test_download_stream import Streamed  # archive answers in the stream=True form (P9)

# Run A ships FILES_A, run B FILES_B: one function per file (no top-level code), one import the graph build reads from disk.
FILES_A = {"app.py": "def master_app():\n    import util\n    return util.helper()\n",
           "util.py": "def helper():\n    total = sum(len(str(i)) for i in range(9))\n    return total\n"}
FILES_B = {"main.py": "def develop_main():\n    import tools\n    return tools.run()\n",
           "tools.py": "def run():\n    count = len([i for i in range(9) if i % 2 == 0])\n    return count\n"}
MARK, FILE, COMMIT = {"A": "master_app", "B": "develop_main"}, {"A": "app.py", "B": "main.py"}, "0123456789abcdef" * 2 + "01234567"

def zip_of(top, files):  # GitHub archive layout: every file under one top folder
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        [zf.writestr(f"{top}/{name}", text) for name, text in files.items()]
    return buf.getvalue()
class FakeGitHub:  # api.github.com + github.com archives behind requests.get; zips keyed by ref (None = any); gates per repo
    def __init__(self, zips, sha="f" * 40):
        self.zips, self.sha, self.status, self.archives, self.gates, self.at = zips, sha, 200, [], {}, {}
    def hold(self, repo):  # the archive download of `repo` marks at[repo], then waits until gates[repo] is set
        self.gates[repo], self.at[repo] = threading.Event(), threading.Event()
    def get(self, url, params=None, **kwargs):
        tail, ref = url.rsplit("/", 1)[1], (params or {}).get("sha")
        if url.startswith("https://api.github.com/"):
            sha = hashlib.sha1(ref.encode()).hexdigest() if ref else (tail if len(tail) == 40 else self.sha)
            return SimpleNamespace(status_code=200, json=lambda: [{"sha": sha}] if tail == "commits" else {"sha": sha})
        repo, ref = url.split("/")[4], url.split("/archive/")[1].removeprefix("refs/heads/").removesuffix(".zip")
        self.archives.append(url)
        if repo in self.gates:
            self.at[repo].set(), self.gates[repo].wait(10)
        z = self.zips.get(ref, self.zips.get(None)) if self.status == 200 else None
        return Streamed(404 if z is None else 200, [z or b"Not Found"])

def top_with(tmp, name):  # the top-level folder under TMP_DIR holding a file `name`, or None
    return next((os.path.relpath(d, tmp).split(os.sep)[0] for d, _, fs in os.walk(tmp) if name in fs), None)
def bg(fn, *args, **kwargs):  # run fn on its own started thread; the box gets its return value
    box, t = [], threading.Thread(target=lambda: box.append(fn(*args, **kwargs)))
    return t.start() or t, box

class Boundary:  # wraps FakeTextEmbedding.embed and .passage_embed; tells runs apart by chunk text; records each run's top folder
    def __init__(self, monkeypatch, tmp, plan):  # plan: run -> "pass" | "hold" (wait for go) | "raise" (model crashes)
        self.top, self.at, self.go = {}, {r: threading.Event() for r in plan}, {r: threading.Event() for r in plan}
        cls, orig = fastembed.TextEmbedding, fastembed.TextEmbedding.passage_embed
        def embed(model, documents, **kwargs):
            docs = list(documents)
            for run in [r for r in plan if any(MARK[r] in d for d in docs)]:
                self.top[run], _ = top_with(tmp, FILE[run]), self.at[run].set()
                if plan[run] == "raise":
                    raise RuntimeError("embedding model crashed")
                self.go[run].wait(10 if plan[run] == "hold" else 0)
            return orig(model, docs, **kwargs)
        [monkeypatch.setattr(cls, name, embed) for name in ("embed", "passage_embed")]

def call_ingest(repo="fastapi", ref=None):  # one ingest through the route function, async today, plain def after B6b
    from api.routes.ingest import IngestRequest, ingest_repo
    body = {} if ref is None else {"commit_hash": ref} if len(ref) == 40 else {"branch": ref}
    try:
        r = ingest_repo(IngestRequest(github_url=f"https://github.com/o/{repo}", **body))
        return asyncio.run(r) if inspect.iscoroutine(r) else r
    except HTTPException as e:
        return e

@pytest.fixture
def world(harness, monkeypatch, tmp_path):
    harness.reload(env={"TMP_DIR": str(tmp_path)})
    monkeypatch.setattr("core.ingestion.cloner.TMP_DIR", str(tmp_path))
    def make(refs, plan):  # refs: (ref of run A, ref of run B)
        monkeypatch.setattr(requests, "get", FakeGitHub(dict(zip(refs, (zip_of("x-a", FILES_A), zip_of("x-b", FILES_B))))).get)
        return Boundary(monkeypatch, str(tmp_path), plan)
    return make

@pytest.mark.parametrize("repo, ref", [("fastapi", "master"), ("flask", "develop")])
def test_single_ingest_clones_into_fresh_folder_and_removes_it(world, tmp_path, repo, ref):
    # Break: clone reuses TMP_DIR/<repo> and wipes what is there (cloner.py:268-275); or clean-up skipped (ingest.py:480).
    (tmp_path / repo).mkdir() or (tmp_path / repo / "left.py").write_text("x = 1\n")  # stray folder of an earlier crash
    b = world((ref,), {"A": "pass"})
    r = call_ingest(repo, ref)
    assert (r.files_indexed, (tmp_path / b.top["A"]).exists()) == (2, False)  # own folder gone at the end
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == [repo, f"{repo}/left.py"]  # stray kept

def both_held(world, tmp_path, refs):  # A clones, holds at embed; B clones, holds; snapshot; free A, A ends; free B, B ends
    b, runs, snap = world(refs, {"A": "hold", "B": "hold"}), [], {}
    try:
        for run, ref in zip("AB", refs):
            runs.append(bg(call_ingest, ref=ref))
            assert b.at[run].wait(5), f"run {run} never reached the embed step"
        snap = {r: top_with(str(tmp_path), FILE[r]) for r in "AB"}
        b.go["A"].set(), runs[0][0].join(10)
    finally:
        [(e.set(), t.join(10)) for e, (t, _) in zip(b.go.values(), runs)]
    return [box[0] for _, box in runs], b.top, snap

@pytest.mark.parametrize("refs", [("master", "develop"), ("main", "feature/x")])
def test_same_repo_two_branches_at_once_both_index_their_own_files(world, tmp_path, refs):
    # Break (G1): shared TMP_DIR/fastapi: B's clone wipes A's files before A's graph reads disk, A's clean-up deletes B's.
    from api.routes.ingest import graph_cache
    got = [(r.files_indexed, r.chunks_stored, sorted(graph_cache[r.repo_name]["graph"].nodes),
            sorted(graph_cache[r.repo_name]["graph"].edges)) for r in both_held(world, tmp_path, refs)[0]]
    assert got == [(2, 2, ["app.py", "util.py"], [("app.py", "util.py")]), (2, 2, ["main.py", "tools.py"], [("main.py", "tools.py")])]

@pytest.mark.parametrize("refs", [("master", "develop"), ("master", COMMIT)])
def test_two_runs_of_one_repo_name_get_different_folders(world, tmp_path, refs):
    # Break: the folder is named after the repo only (cloner.py:269), so both runs get TMP_DIR/fastapi and B wipes A's files.
    _, top, snap = both_held(world, tmp_path, refs)
    assert (top["A"] != top["B"], snap) == (True, {"A": top["A"], "B": top["B"]})  # both folders live while both held

@pytest.mark.parametrize("ref_dead, ref_held", [("master", "develop"), ("develop", "master")])
def test_run_failing_at_embed_removes_only_its_own_folder(world, tmp_path, ref_dead, ref_held):
    # Break: no clean-up on the error path (ingest.py:480); or the dead run's clone wiped the held run's TMP_DIR/<repo>.
    b = world((ref_dead, ref_held), {"A": "raise", "B": "hold"})
    held, _ = bg(call_ingest, ref=ref_held)
    try:
        assert b.at["B"].wait(5), "held run never reached the embed step"
        dead = call_ingest(ref=ref_dead)
        assert (getattr(dead, "status_code", None), (tmp_path / b.top["A"]).exists()) == (500, False)  # own folder gone
        assert top_with(str(tmp_path), FILE["B"]) == b.top["B"]  # the held run's folder still holds its files
    finally:
        b.go["B"].set(), held.join(10)
