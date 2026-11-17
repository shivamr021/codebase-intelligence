"""tests/test_chunker.py — B2a module chunks for top-level code (seam A: chunk_file on files in tmp_path)."""
import pytest
from core.embeddings.embedder import _chunk_id
from core.ingestion.chunker import chunk_file

LANG = {"py": "python", "ts": "typescript", "js": "javascript"}

def chunks_of(tmp_path, name, text, newline="\n"):
    path = tmp_path / name
    path.write_bytes(text.replace("\n", newline).encode())
    return chunk_file({"path": str(path), "rel_path": f"src/{name}", "language": LANG[name.rsplit(".", 1)[1]], "size": 1})

def modules(chunks):
    return [c for c in chunks if c["chunk_type"] == "module"]

def holder(chunks, needle, line):  # the module chunk holding `needle` at file line `line`, its text line agreeing
    hits = [c for c in modules(chunks) if c["start_line"] <= line <= c["end_line"]
            and c["text"].split("\n")[line - c["start_line"]].startswith(needle)]
    assert hits, f"no module chunk holds {needle!r} at line {line}: {[(c['chunk_type'], c['start_line'], c['end_line']) for c in chunks]}"
    return hits[0]

FN = "def {0}(items):\n    total = sum(len(str(item)) for item in items)\n    return total\n"
PY_HEAD = 'import os\n\nEMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")\n\n\n' + FN.format("load")
PY_MID = FN.format("load") + '\n\nDEFAULT_TOP_K = 5\nEMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")\n\n\n' + FN.format("save")

@pytest.mark.parametrize("text, line", [pytest.param(PY_HEAD, 3, id="above-functions"), pytest.param(PY_MID, 7, id="between-functions")])
def test_python_settings_line_lands_in_a_module_chunk_with_its_line_number(tmp_path, text, line):
    # Break: top-level code outside functions is never chunked (today), or gets the wrong line numbers.
    holder(chunks_of(tmp_path, "config.py", text), "EMBEDDING_MODEL =", line)

TS = ('export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";\n'
      'export const REQUEST_TIMEOUT_MS = 30000;\n\n'
      'export interface IngestRequest {\n  github_url: string;\n  branch?: string;\n}\n\n'
      'export interface IngestResponse {\n  repo_name: string;\n  chunks_stored: number;\n}\n')
JS = ('const DEFAULT_TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 30000);\nconst RETRY_LIMIT = 3;\n\n'
      'function fetchWithRetry(url) {\n  return fetch(url, { signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS) });\n}\n')

@pytest.mark.parametrize("name, text, first, second", [
    pytest.param("api.ts", TS, "export const API_BASE =", "export const REQUEST_TIMEOUT_MS", id="ts-above-interfaces"),
    pytest.param("http.js", JS, "const DEFAULT_TIMEOUT_MS =", "const RETRY_LIMIT", id="js-above-function"),
])
def test_top_level_consts_give_one_module_chunk_over_lines_1_to_2(tmp_path, name, text, first, second):
    # Break: no module chunk (today); or it swallows the interfaces/function below; or lines 1-2 are split apart.
    chunk = holder(chunks_of(tmp_path, name, text), first, 1)
    assert (chunk["start_line"], chunk["text"].split("\n")[1].startswith(second)) == (1, True)
    assert chunk["end_line"] in (2, 3) and "interface" not in chunk["text"] and "function" not in chunk["text"]

def test_file_of_only_functions_gets_no_module_chunk(tmp_path):
    # GUARD (green today, no module chunks exist): bites after B2a if gaps under MIN_CHUNK_CHARS (`import os`, blanks) become chunks.
    chunks = chunks_of(tmp_path, "util.py", "import os\n\n" + "\n\n".join(FN.format(n) for n in ("a", "b", "c")))
    assert sorted((c["chunk_type"], c["start_line"]) for c in chunks) == [("function", 3), ("function", 8), ("function", 13)]

def test_120_line_gap_splits_into_chunks_of_at_most_50_lines_losing_no_line(tmp_path):
    # Break: no module chunk (today); one 120-line chunk; a split that drops or misnumbers the lines at a 50-line seam.
    consts = [f'SETTING_{i:03d} = "value-{i:03d}"' for i in range(1, 121)]  # file lines 4..123
    chunks = chunks_of(tmp_path, "settings.py", FN.format("first") + "\n".join(consts) + "\n" + FN.format("last"))
    for c in modules(chunks):
        assert c["end_line"] - c["start_line"] + 1 <= 50
    for line, const in enumerate(consts, start=4):
        holder(chunks, const, line)

def test_class_methods_stay_out_of_the_module_chunk_for_a_constant_below(tmp_path):
    # Break: no module chunk (today); the gap walk descends into the class body and repeats its attribute or method text.
    cls = ("class Store:\n    DEFAULT_TTL_SECONDS = 3600  # entries expire after one hour of idle time\n\n"
           "    def get(self, key):\n        return self.data.get(key, None) or self.load(key)\n\n"
           "    def put(self, key, value):\n        self.data[key] = value\n        return self.flush(key)\n")
    chunks = chunks_of(tmp_path, "store.py", cls + '\n\nCACHE_DIR = "/var/cache/store/default-namespace/objects-v2"\n')
    assert len(modules(chunks)) == 1, [(c["chunk_type"], c["start_line"], c["end_line"]) for c in chunks]
    text = holder(chunks, "CACHE_DIR =", 12)["text"]
    assert all(s not in text for s in ("DEFAULT_TTL", "def get", "def put", "return self", "class Store"))

def test_file_without_functions_keeps_the_line_fallback_and_nothing_else(tmp_path):
    # Break: fallback skipped (`chunker.py:373` -> `if False:`, returns []); after B2a, module chunks duplicating the raw ones.
    script = "".join(f'print("bootstrap step {i:02d} of the setup script runs now")\n' for i in range(1, 61))
    chunks = chunks_of(tmp_path, "setup.py", script)
    assert [(c["chunk_type"], c["start_line"], c["end_line"]) for c in chunks] == [("raw", 1, 50), ("raw", 51, 60)]

def test_module_chunks_of_one_file_get_unique_ids(tmp_path):
    # Break: no module chunks (today); every module chunk of a file shares one name and start line, so ids collide on upsert.
    gap = 'RETRY_DELAYS_SECONDS = [1, 2, 4, 8, 16, 32]\nUSER_AGENT = "codebase-intelligence/1.0"\n'
    chunks = chunks_of(tmp_path, "jobs.py", FN.format("a") + gap + FN.format("b") + gap + FN.format("c"))
    ids = [_chunk_id("repo", c["file_path"], c["name"], c["start_line"]) for c in chunks]
    assert len(modules(chunks)) >= 2 and len(set(ids)) == len(ids) == len(modules(chunks)) + 3

def test_module_heavy_file_gets_exactly_one_module_chunk_per_gap(tmp_path):
    # Break (G9): no module chunks (today); one chunk per line or per statement, blowing up the 1000-chunk cap.
    block = 'BLOCK_{0}_HOST = "service-{0}.internal.example"\nBLOCK_{0}_PORT = 8000\nBLOCK_{0}_PATH = "/api/v1/{0}"\n'
    text = FN.format("f0") + "".join(block.format(i) + FN.format(f"f{i}") for i in range(1, 10))
    chunks = chunks_of(tmp_path, "heavy.py", text)
    assert len(modules(chunks)) == 9 and all(c["end_line"] - c["start_line"] + 1 <= 50 for c in modules(chunks))
    for i in range(1, 10):  # gap i sits on file lines 6i-2 .. 6i
        assert holder(chunks, f"BLOCK_{i}_HOST =", 6 * i - 2) is holder(chunks, f"BLOCK_{i}_PATH =", 6 * i)

def test_syntax_error_file_with_one_function_keeps_its_top_level_text(tmp_path):
    # Break: the function chunk makes `chunks` non-empty, skipping the fallback at `chunker.py:373`, so line 1 is lost (today).
    text = ('API_TIMEOUT_SECONDS = int(os.environ.get("API_TIMEOUT_SECONDS", "30"))\n\n'
            'def fetch_page(session, url):\n    response = session.get(url, timeout=API_TIMEOUT_SECONDS)\n    return response.text\n\n'
            'def broken(:\n    pass\n')
    chunks = chunks_of(tmp_path, "client.py", text)
    assert ("function", "fetch_page", 3) in [(c["chunk_type"], c["name"], c["start_line"]) for c in chunks]
    holder(chunks, "API_TIMEOUT_SECONDS =", 1)

def test_crlf_file_keeps_line_numbers_of_module_and_function_chunks(tmp_path):
    # Break: no module chunk (today); a `\r\n` counted as two line breaks shifts every line after it; stray `\r` in text.
    chunks = chunks_of(tmp_path, "config.py", PY_MID, newline="\r\n")
    assert sorted((c["start_line"], c["end_line"]) for c in chunks if c["chunk_type"] == "function") == [(1, 3), (10, 12)]
    assert "\r" not in holder(chunks, "EMBEDDING_MODEL =", 7)["text"]
