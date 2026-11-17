"""tests/test_llm_failures.py — B3a honest LLM failures, seams A call_llm, B /bugs, C /ingest; Groq SDK client + requests.get faked."""
import io, zipfile  # noqa: E401
from types import SimpleNamespace
import httpx, openai, pytest, requests  # noqa: E401
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

SHA1, SHA2 = "a1" * 20, "b2" * 20
FILES = {"app.py": "from util import helper\n\ndef main():\n    return helper()\n", "util.py": "def helper():\n    return 1\n"}
GOOD_ARCH = "SUMMARY\nApp calls util.\n```mermaid\nflowchart TD\nA-->B\n```"
NOT_FOUND = openai.NotFoundError(  # what Groq sends for a retired model id
    "Error code: 404 - The model `llama-3.3-70b-versatile` does not exist", body=None,
    response=httpx.Response(404, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")))

class Groq:  # the Groq server: each request takes the next scripted reply: text, (content, finish_reason), or an error
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
    def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        choice = Choice(index=0, finish_reason=finish, message=ChatCompletionMessage(role="assistant", content=content))
        return ChatCompletion(id="chatcmpl-1", object="chat.completion", created=0, model=kwargs["model"], choices=[choice])

def groq(monkeypatch, *replies):
    monkeypatch.setattr("core.llm.client._client", server := Groq(replies))
    return server

def github(monkeypatch, sha, files=FILES):  # GitHub: commit lookup answers `sha`, archive answers the zip; logs URLs
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(f"demo-main/{name}", text)
    urls = []
    def get(url, **kwargs):
        urls.append(url)
        return SimpleNamespace(status_code=200, json=lambda: [{"sha": sha}], content=buf.getvalue())
    monkeypatch.setattr(requests, "get", get)
    return urls

def stored(harness, name):  # the raw metadata record in the shared store; {} when none was saved
    points = harness.store.scroll("repo_metadata", limit=100)[0]
    return next((p.payload for p in points if p.payload["repo_name"] == name), {})

@pytest.fixture
def api(harness, monkeypatch, tmp_path):
    harness.reload("core.ingestion.cloner", env={"TMP_DIR": str(tmp_path)})  # env route once B5 lands; the setattr today
    monkeypatch.setattr("core.ingestion.cloner.TMP_DIR", str(tmp_path), raising=False)
    from core.storage.repo_metadata import ensure_metadata_collection
    ensure_metadata_collection()  # the store already holds the metadata collection; the empty-store 500 is B1b's case
    from main import app
    with TestClient(app) as client:
        yield client

def ingest(api):
    return api.post("/api/v1/ingest", json={"github_url": "https://github.com/o/demo"})

def bugs(api, harness):
    from core.embeddings.embedder import embed_and_store
    embed_and_store(harness.chunks, "demo")
    return api.get("/api/v1/bugs", params={"repo_name": "demo"})

TWO_BUGS = ('[{"file": "a.py", "line": 3, "severity": "high", "issue": "x", "suggestion": "y"}, '
            '{"file": "b.py", "line": 9, "severity": "low", "issue": "z", "suggestion": "w"}]')

@pytest.mark.parametrize("reply", [TWO_BUGS, "Here are the bugs:\n" + TWO_BUGS + "\nDone."], ids=["bare", "prose-around"])
def test_bug_array_returns_200_with_each_bug(api, harness, monkeypatch, reply):
    # middle, existing. Break: bugs.py:55 `bugs = []` (parse drops the array) -> 0 bugs at 200.
    groq(monkeypatch, reply)
    r = bugs(api, harness)
    assert (r.status_code, [(b["file"], b["line"]) for b in r.json()["bugs"]], r.json()["total"]) == (200, [("a.py", 3), ("b.py", 9)], 2)

def test_valid_empty_bug_array_is_an_honest_200(api, harness, monkeypatch):
    # edge, existing. Break: bugs.py:61 returns `[{}]` when the list is empty -> one made-up bug, total 1.
    groq(monkeypatch, "[]")
    r = bugs(api, harness)
    assert (r.status_code, r.json()["bugs"], r.json()["total"], "detail" in r.json()) == (200, [], 0, False)

@pytest.mark.parametrize("reply", [
    "I reviewed the code and everything looks fine to me.",  # no array: bugs.py:50-52
    '[{"file": "a.py", "line": 3, "tags": ["x"]}, {"file": "b.py", "li',  # cut-off array with an inner ]: bugs.py:63-66
    NOT_FOUND,  # Groq 404: client.py:105-108 -> "[LLM Error] Groq API error: ..."
], ids=["prose-no-array", "truncated-array", "groq-404"])
def test_unusable_bug_reply_returns_502_with_a_reason(api, harness, monkeypatch, reply):
    # bad, new. Break: _parse_bug_json returns [] on failure and the route answers 200 "no bugs" (bugs.py:52, :66, :98).
    groq(monkeypatch, reply)
    r = bugs(api, harness)
    assert (r.status_code, bool(str(r.json().get("detail") or "").strip())) == (502, True)

def test_empty_reply_cut_by_token_limit_is_an_llm_error(monkeypatch):
    # edge, new. Break: client.py:97 returns "".strip() as a normal answer -> callers treat "" as real content.
    from core.llm.client import call_llm
    server = groq(monkeypatch, ("", "length"))
    assert (call_llm("explain")[:11], len(server.requests)) == ("[LLM Error]", 1)

@pytest.mark.parametrize("finish, want", [("length", "[LLM Error] Empty reply from the model (finish_reason=length)"),
                                          ("stop", "[LLM Error] Empty reply from the model (finish_reason=stop)")],
                         ids=["length", "stop"])
def test_none_reply_gives_the_exact_empty_reply_message(monkeypatch, finish, want):
    # edge, new (G10). Break: `.strip()` on None -> generic "[LLM Error] Unexpected error: ..." with no finish_reason.
    from core.llm.client import call_llm
    groq(monkeypatch, (None, finish))
    assert call_llm("explain") == want

def test_architecture_reply_with_mermaid_is_saved_with_its_body(api, harness, monkeypatch):
    # middle, existing. Break: architecture.py:23 `mermaid = ""` -> the "Try Re-ingesting" fallback chart is saved instead.
    groq(monkeypatch, GOOD_ARCH)
    github(monkeypatch, SHA1)
    record = stored(harness, ingest(api).json()["repo_name"])
    assert (record.get("mermaid"), "App calls util." in record.get("summary", "")) == ("flowchart TD\nA-->B", True)

@pytest.mark.parametrize("reply, check_message", [
    (("", "length"), False),  # edge, new: empty reply at the token limit
    (NOT_FOUND, False),  # bad, existing: architecture.py:68-72 + ingest.py:159
    ("SUMMARY\nApp calls util.", True),  # bad, new: summary with no mermaid block
], ids=["empty-length", "groq-404", "no-mermaid"])
def test_failed_architecture_is_not_saved(api, harness, monkeypatch, reply, check_message):
    # Break: architecture.py:68 `if False:` (404 text saved as summary); "" and no-mermaid replies get the fallback
    # chart (architecture.py:76-81) and are saved. "Not saved" = the record holds no summary and no mermaid.
    groq(monkeypatch, reply)
    github(monkeypatch, SHA1)
    body = ingest(api).json()
    record = stored(harness, body["repo_name"])
    said_failed = "architecture" in body["message"].lower() and "fail" in body["message"].lower()
    assert (record.get("summary") or "", record.get("mermaid") or "", said_failed or not check_message) == ("", "", True)
