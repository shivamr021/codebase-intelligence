"""tests/test_tmp_dir.py — B5: TMP_DIR from env through clone_repo (seam F); requests faked at the HTTP boundary."""
import io, zipfile  # noqa: E401
import pytest
import requests
from test_download_stream import Streamed  # archive answers in the stream=True form (P9)

@pytest.mark.parametrize("http, sub, want", [
    pytest.param(200, "x", ("cloned", ["print('hi')\n"]), id="middle-clone-lands-under-tmp-dir"),
    pytest.param(404, "not/yet", ("error", []), id="bad-missing-tmp-dir-created-even-on-failed-download"),
])
def test_clone_uses_tmp_dir_from_env_and_creates_it(harness, monkeypatch, tmp_path, http, sub, want):
    # Break: TMP_DIR hardcoded in config.py, copied by cloner.py at import, or makedirs(TMP_DIR) dropped.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("demo-main/app.py", "print('hi')\n")
    monkeypatch.setattr(requests, "get", lambda url, **kw: Streamed(http, [buf.getvalue()]))
    harness.reload("core.ingestion.cloner", env={"TMP_DIR": str(tmp_path / sub)})
    from core.ingestion.cloner import clone_repo
    status = clone_repo("https://github.com/o/demo", "o", "demo")["status"]
    files = [p.read_text() for p in (tmp_path / sub).rglob("app.py")]
    assert (status, (tmp_path / sub).is_dir(), files) == (want[0], True, want[1])
