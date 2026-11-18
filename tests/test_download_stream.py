"""tests/test_download_stream.py — P9 streamed download (seam A: clone_repo, requests.get faked like a stream=True response)."""
import io, os, tempfile, zipfile  # noqa: E401
import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

def zip_of(top, files):  # GitHub archive layout: every file under one top folder
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(f"{top}/{name}", text)
    return buf.getvalue()

ZIP = zip_of("repo-abc", {"src/a.py": "x = 1\n"})

class Streamed:  # requests.Response as stream=True returns it: pieces from iter_content; .content reads them all, like the real one
    def __init__(self, status_code, pieces, content_ok=True):
        self.status_code, self.headers, self.pieces, self.content_ok = status_code, {"Content-Type": "application/zip"}, pieces, content_ok
    def iter_content(self, chunk_size=1, decode_unicode=False):
        for piece in self.pieces:
            if isinstance(piece, Exception):
                raise piece
            piece = piece() if callable(piece) else piece
            step = chunk_size or len(piece) or 1  # the real one cuts each read to chunk_size; None = as it arrives
            yield from (piece[i:i + step] for i in range(0, len(piece), step))
    @property
    def content(self):
        assert self.content_ok, "whole body read through .content"
        return b"".join(self.iter_content())
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")
    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.close()

@pytest.fixture
def dirs(harness, monkeypatch, tmp_path):  # (TMP_DIR, the system temp folder), both empty and watched
    tmp_dir, sys_tmp = tmp_path / "tmp_dir", tmp_path / "sys_tmp"
    sys_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sys_tmp))
    harness.reload("core.ingestion.cloner", env={"TMP_DIR": str(tmp_dir)})
    return tmp_dir, sys_tmp

def files_in(d):
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.exists() else []

def serve(monkeypatch, response, want_stream=False):  # every requests.get answers `response`; optionally insists on stream=True
    def get(url, **kwargs):
        if want_stream and kwargs.get("stream") is not True:
            raise AssertionError(f"requests.get without stream=True buffers the whole body: {kwargs}")
        return response
    monkeypatch.setattr(requests, "get", get)

def clone():
    from core.ingestion.cloner import clone_repo
    return clone_repo("https://github.com/o/repo", "o", "repo", commit_hash="abc" * 13 + "a")

def test_archive_in_pieces_is_written_inside_tmp_dir(dirs, monkeypatch):
    # Break: zip written to the system temp folder (NamedTemporaryFile without dir=TMP_DIR, cloner.py:334).
    tmp_dir, sys_tmp = dirs
    seen = []  # (files under TMP_DIR, files in system temp) as each piece arrives
    def piece(part):
        return lambda: seen.append((len(files_in(tmp_dir)), len(files_in(sys_tmp)))) or part
    third = len(ZIP) // 3
    serve(monkeypatch, Streamed(200, [piece(ZIP[:third]), piece(ZIP[third:2 * third]), piece(ZIP[2 * third:])]))
    got = clone()
    text = open(os.path.join(got["local_path"], "src", "a.py")).read()
    assert (got["status"], text, seen) == ("cloned", "x = 1\n", [(1, 0), (1, 0), (1, 0)])

def test_whole_body_is_never_held_at_once(dirs, monkeypatch):
    # Break: body read via response.content (cloner.py:335), or stream=True forgotten so requests buffers it at get (:317).
    serve(monkeypatch, Streamed(200, [ZIP[:100], ZIP[100:]], content_ok=False), want_stream=True)
    got = clone()
    text = open(os.path.join(got.get("local_path") or "missing", "src", "a.py")).read() if got["status"] == "cloned" else got["message"]
    assert (got["status"], text) == ("cloned", "x = 1\n")

def test_connection_drop_midway_leaves_no_half_file(dirs, monkeypatch):
    # Break: half file left on disk (zip path recorded only after the write loop, cloner.py:334-336), or run folder kept.
    tmp_dir, sys_tmp = dirs
    drop = requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead(100 bytes read, 900 more expected)")
    serve(monkeypatch, Streamed(200, [ZIP[:100], drop]))
    got = clone()
    assert (got["status"], files_in(tmp_dir), files_in(sys_tmp), [p.name for p in tmp_dir.iterdir()]) == ("error", [], [], [])

def test_read_timeout_while_pieces_arrive_reports_timeout(dirs, monkeypatch):
    # Break: iter_content wraps a read timeout in ConnectionError (requests/models.py:942-943), which falls to "Unexpected error" (cloner.py:367-373).
    late = requests.exceptions.ConnectionError(ReadTimeoutError(None, "https://codeload.github.com", "Read timed out. (read timeout=60)"))
    serve(monkeypatch, Streamed(200, [ZIP[:100], late]))
    got = clone()
    assert (got["status"], got["message"]) == ("error", "Download timed out after 60s. Repository may be too large.")

def test_ok_answer_that_is_not_a_zip_leaves_no_temp_file(dirs, monkeypatch):
    # Break: temp zip kept after a failed extract (delete skipped at cloner.py:376).
    tmp_dir, sys_tmp = dirs
    serve(monkeypatch, Streamed(200, [b"<!DOCTYPE html><html><body>", b"Sign in to GitHub</body></html>"]))
    got = clone()
    assert (got["status"], files_in(tmp_dir), files_in(sys_tmp)) == ("error", [], [])
