"""tests/test_ask_honest.py — P3 seam B: POST /api/v1/ask with fake Groq + Qdrant :memory: reports errors, cut-offs,
refusals and sources honestly. New fields are read with .get() so a missing field is an assertion diff, not a KeyError."""
import httpx, openai  # noqa: E401
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

REFUSAL = "I cannot find enough information in the indexed code to answer this."
CHUNKS = [{"file_path": "a.py", "name": "alpha", "text": "def alpha():\n    return 1\n", "chunk_type": "function",
           "start_line": 1, "end_line": 3, "language": "python"},
          {"file_path": "b.py", "name": "beta", "text": "def beta():\n    x = 2\n\n\n    return x\n", "chunk_type": "function",
           "start_line": 4, "end_line": 9, "language": "python"}]

def reply_with(harness, content, finish="stop"):  # the Groq server's next reply; an Exception is raised instead
    def create(**kwargs):
        harness.groq.requests.append(kwargs)
        if isinstance(content, Exception):
            raise content
        choice = Choice(index=0, finish_reason=finish, message=ChatCompletionMessage(role="assistant", content=content))
        return ChatCompletion(id="chatcmpl-1", object="chat.completion", created=0, model=kwargs["model"], choices=[choice])
    harness.groq.chat.completions.create = create

def ask(harness, store=True):
    from core.embeddings.embedder import embed_and_store
    from main import app
    if store:
        embed_and_store(CHUNKS, "demo")
    return TestClient(app).post("/api/v1/ask", json={"repo_name": "demo", "question": "Where is alpha defined?"})

def test_ask_lists_each_retrieved_chunk_as_a_source(harness):
    # P3-M3 middle, new. Break: `sources` left empty, or lines parsed from the header text instead of start/end_line.
    reply_with(harness, "alpha is in a.py:2")
    sources = ask(harness).json().get("sources") or []
    got = {(s.get("file"), s.get("name"), s.get("lines"), s.get("score")) for s in sources}  # score 1.0 [measured]
    assert (len(sources), got) == (2, {("a.py", "alpha", "1-3", 1.0), ("b.py", "beta", "4-9", 1.0)})

def test_normal_reply_is_a_plain_success(harness):
    # P3-M4 middle, new. Break: `refused` or the cut-off flag default to true, or are missing (None) on a normal reply.
    reply_with(harness, "alpha is in a.py:2")
    r = ask(harness)
    body = r.json()
    assert (r.status_code, body.get("answer"), body.get("refused"), body.get("cut_off")) == (200, "alpha is in a.py:2", False, False)

def test_old_response_fields_stay(harness):
    # P3-M5 middle, existing. Break: query.py:76 `chunks_used=0`, or new fields replace the old shape.
    reply_with(harness, "alpha is in a.py:2")
    body = ask(harness).json()
    old = {k: body.get(k) for k in ("status", "repo_name", "question", "answer", "chunks_used")}
    assert old == {"status": "success", "repo_name": "demo", "question": "Where is alpha defined?",
                   "answer": "alpha is in a.py:2", "chunks_used": 2}

def test_reply_cut_by_the_token_limit_with_text_is_flagged(harness):
    # P3-E1 edge, new. Break: cut-off detected only when the text is empty (client.py:55) -> partial text poses as whole.
    reply_with(harness, "partial answer", finish="length")
    r = ask(harness)
    assert (r.status_code, r.json().get("answer"), r.json().get("cut_off")) == (200, "partial answer", True)

def test_the_fixed_refusal_sentence_alone_is_a_refusal(harness):
    # P3-E2 edge, new. Break: the refusal is passed off as an answer (`refused` false or missing).
    reply_with(harness, REFUSAL)
    r = ask(harness)
    assert (r.status_code, r.json().get("refused")) == (200, True)

def test_longer_answer_containing_the_refusal_sentence_is_not_a_refusal(harness):
    # P3-E5 edge, new (ruling 5). Break: refusal found by substring instead of exact match -> `refused` true.
    reply_with(harness, REFUSAL + " But `a.py:2` defines alpha.")
    r = ask(harness)
    assert (r.status_code, r.json().get("refused")) == (200, False)

def test_nothing_retrieved_is_404_naming_the_repo_and_never_reaches_the_model(harness):
    # P3-E4 edge, existing. Break: query.py:49 `if False:` -> the empty result is sent to the model and answered.
    reply_with(harness, "made-up answer")
    r = ask(harness, store=False)
    assert (r.status_code, "'demo'" in str(r.json().get("detail")), len(harness.groq.requests)) == (404, True, 0)

def test_groq_api_error_is_502_llm_upstream_error_with_its_reason(harness):
    # P3-B1 bad, new. Break: call_llm returns "[LLM Error] ..." text (client.py:116) and /ask sends it as a 200 answer.
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    reply_with(harness, openai.APIError("upstream boom", request, body=None))
    r = ask(harness)
    body = r.json()
    assert (r.status_code, body.get("error_code"), "upstream boom" in str(body.get("detail"))) == (502, "LLM_UPSTREAM_ERROR", True)

def test_reply_with_no_text_is_502_llm_upstream_error(harness):
    # P3-B2 bad, new. Break: only API errors mapped -> the empty-reply "[LLM Error]" text still answers 200.
    reply_with(harness, None, finish="length")
    r = ask(harness)
    assert (r.status_code, r.json().get("error_code")) == (502, "LLM_UPSTREAM_ERROR")
