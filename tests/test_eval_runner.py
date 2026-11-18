"""tests/test_eval_runner.py — P2 eval runner (seam A: evals.run_eval.main on a fixture repo; fake GitHub zip, X1, Qdrant :memory:, fake Groq)."""
import io, json, zipfile  # noqa: E401
import requests

PIN = "a" * 40
URL = "https://github.com/o/demo"
FILES = {  # 3 files, 4 chunks by hand: a.py 1 function, b.py 2 functions, c.py 1 function; no gap line holds code
    "a.py": "def alpha_count(items):\n    # alpha: count every item once\n    seen = [item for item in items]\n"
            "    total = len(seen)\n    return total\n",
    "b.py": "def bravo_join(parts):\n    joined = \", \".join(str(part) for part in parts)\n    return joined\n\n\n"
            "def charlie_split(text):\n    pieces = [piece.strip() for piece in text.split(\",\")]\n    return pieces\n",
    "c.py": "def delta_scale(values, factor):\n    scaled = [value * factor for value in values]\n    return scaled\n",
}
Q_ALPHA = {"id": "q1", "question": "where is alpha counted", "golds": [{"file_path": "a.py", "start_line": 1, "end_line": 5}]}
Q_BRAVO = {"id": "q2", "question": "how does bravo join parts", "golds": [{"file_path": "b.py", "start_line": 1, "end_line": 3}]}

def reply(url, status, content):  # a real requests.Response, as requests.get returns it
    r = requests.Response()
    r.url, r.status_code, r._content = url, status, content
    return r

class FakeGitHub:  # api.github.com commit lookup + github.com archive at requests.get; serves `sha` whatever was asked
    def __init__(self, files, sha=PIN, archive_status=200):
        self.files, self.sha, self.archive_status, self.urls = files, sha, archive_status, []
    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
        self.urls.append(url)
        if url.startswith("https://api.github.com/repos/o/demo/commits"):
            body = {"sha": self.sha} if "/commits/" in url else [{"sha": self.sha}]
            return reply(url, 200, json.dumps(body).encode())
        if "/archive/" not in url or self.archive_status != 200:
            return reply(url, self.archive_status if "/archive/" in url else 404, b"Not Found")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:  # GitHub layout: top folder <repo>-<sha>/
            for name, text in self.files.items():
                zf.writestr(f"demo-{self.sha}/{name}", text)
        return reply(url, 200, buf.getvalue())

def run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, gold_rows, files=FILES, mode="retrieval", gh=None, extra=()):
    gh = gh or FakeGitHub(files)
    monkeypatch.setattr(requests, "get", gh.get)
    harness.reload(env={"TMP_DIR": str(tmp_path / "work")})
    monkeypatch.setattr("core.ingestion.cloner.TMP_DIR", str(tmp_path / "work"), raising=False)
    model_dir = x2_tokenizer.parent
    (model_dir / "model_optimized.onnx").write_bytes(b"abc")  # the file fastembed loads (onnx_model.py:38); tokenizer.json beside it
    gold, out = tmp_path / "gold.jsonl", tmp_path / "out" / "results.json"
    gold.write_text("".join(json.dumps(row) + "\n" for row in gold_rows))
    out.parent.mkdir()
    from evals.run_eval import main
    argv = ["--repo", URL, "--pin", PIN, "--gold", str(gold), "--mode", mode, "--out", str(out), "--model-dir", str(model_dir), *extra]
    try:
        code = main(argv)
    except SystemExit as e:
        code = e.code
    return code, (json.loads(out.read_text()) if out.exists() else None), sorted(p.name for p in out.parent.iterdir())

def rows(results):
    return {r["id"]: r for r in results["questions"]}

def groq_replies(harness, monkeypatch, fail_on=None):  # fake Groq server: "see `a.py:3`", or APIError when the prompt holds `fail_on`
    import httpx, openai  # noqa: E401
    real_create = harness.groq._create
    def create(**kwargs):
        if fail_on and fail_on in kwargs["messages"][-1]["content"]:
            harness.groq.requests.append(kwargs)
            raise openai.APIError("upstream 500", httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"), body=None)
        out = real_create(**kwargs)
        out.choices[0].message.content = "see `a.py:3`"
        return out
    monkeypatch.setattr(harness.groq.chat.completions, "create", create)

def test_retrieval_mode_writes_one_scored_row_per_question(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer):
    # Break: runner scores the wrong collection (every gold a miss) or writes only totals (no per-question rows).
    code, results, _ = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [Q_ALPHA, Q_BRAVO])
    assert (code, [(r["id"], r["class"], r["rank"]) for r in results["questions"]]) == (0, [("q1", "HIT", 1), ("q2", "HIT", 1)])

def test_answer_mode_asks_groq_three_times_and_stores_citation_precision(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer):
    # Break: one Groq call instead of 3 (measure.md:182), or the citation score never stored on the row.
    groq_replies(harness, monkeypatch)
    code, results, _ = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [Q_ALPHA], mode="answer")
    assert (code, len(harness.groq.requests), rows(results)["q1"]["precision"]) == (0, 3, 1.0)

def test_every_run_records_chunk_count_and_model_file_sha256(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer):
    # Break: stats left out, or sha256 taken of the wrong file (tokenizer.json sits beside the model file).
    code, results, _ = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [Q_ALPHA])
    stats = results["stats"]  # sha256(b"abc") is the FIPS 180-2 test vector [doc]
    assert (code, stats["chunks"], stats["model_sha256"]) == (0, 4, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

def test_gold_file_the_walk_skips_is_not_indexed(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer):
    # Break: a file the walk never indexed (test_ prefix, walker.py:108) is reported as OUTRANKED.
    files = {**FILES, "test_utils.py": "def echo_helper(value):\n    # echo the value back for the tests\n    return value\n"}
    gold = {"id": "q1", "question": "where is echo used", "golds": [{"file_path": "test_utils.py", "start_line": 1, "end_line": 3}]}
    code, results, _ = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [gold], files=files)
    assert (code, rows(results)["q1"]["class"], rows(results)["q1"]["rank"]) == (0, "NOT_INDEXED", None)

def test_gold_indexed_but_ranked_21_is_outranked(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer):
    # Break: rank cut at 5 (or 20), so an indexed gold past rank 20 reads as NOT_INDEXED.
    # 25 chunks. X1 cosine vs query "kilo": 20 kilo-only chunks 1.0 > gold (kilo + lima) 2/sqrt(6)=0.816 > 4 mike chunks 0.5 -> rank 21.
    fn = "def {}_{:02d}(value):\n    result = value + {}\n    return result\n"
    files = {"k.py": "\n\n".join(fn.format("kilo", i, i) for i in range(1, 21)), "m.py": "\n\n".join(fn.format("mike", i, i) for i in range(1, 5)),
             "g.py": "def kilo_lima_total(value):\n    result = value * 3\n    return result\n"}
    gold = {"id": "q1", "question": "where is kilo handled", "golds": [{"file_path": "g.py", "start_line": 1, "end_line": 3}]}
    code, results, _ = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [gold], files=files)
    assert (code, results["stats"]["chunks"], rows(results)["q1"]["class"]) == (0, 25, "OUTRANKED")

def test_min_chunk_rule_drop_count_and_gold_lines_inside_dropped_pieces(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer):
    # Break: drop count never measured (AD5 step). f and g are 17 chars (< MIN_CHUNK_CHARS 50, chunker.py:126); nested, so no gap lines.
    files = {"e.py": "def outer_with_helpers(items):\n    def f(): return 1\n    def g(): return 2\n    return f() + g() + len(items)\n"}
    gold = {"id": "q1", "question": "where is golf", "golds": [{"file_path": "e.py", "start_line": 1, "end_line": 2}]}  # line 2 only is in f
    code, results, _ = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [gold], files=files)
    assert (code, results["min_chunk_drop"]) == (0, {"dropped": 2, "gold_lines_in_dropped": 1})

def test_premise_pilot_known_bad_index_scores_worse(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer):
    # Break: a metric that cannot drop. Known-bad = each chunk cut to its first line (range and text):
    # gold line 3 of each 5-line function no longer overlaps -> Recall@5 1.0 -> 0.0; lines 2-5 of 3 functions missing = 12.
    body = "def {}(values):\n    total = 0\n    for value in values:\n        total += value\n    return total\n"
    names = ("one", "two", "three")
    files = {f"f_{n}.py": body.format(f"foxtrot_{n}") for n in names}
    golds = [{"id": f"q_{n}", "question": f"where is foxtrot {n}", "golds": [{"file_path": f"f_{n}.py", "start_line": 3, "end_line": 3}]}
             for n in names]
    code, results, _ = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, golds, files=files, extra=["--pilot"])
    assert (code, results["pilot"]) == (0, {"recall@5": {"real": 1.0, "known_bad": 0.0}, "missing_lines": {"real": 0, "known_bad": 12}})

def test_token_fit_counts_stored_text_and_reports_the_one_chunk_over_600(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer):
    # Break: token fit reads the chunker's own counter, not the stored text. X2 counts: big() 700 words + 2 special = 702 > 600;
    # small() 9 words + 2 = 11. big() is ~1.4k chars, under the 3000-char cut (chunker.py:122), so its stored text is whole.
    files = {"big.py": "def big():\n    \"\"\"" + " ".join(["w"] * 698) + "\"\"\"\n",
             "small.py": "def small(value):\n    result = value + 1\n    return result\n"}
    gold = {"id": "q1", "question": "where is small", "golds": [{"file_path": "small.py", "start_line": 1, "end_line": 3}]}
    code, results, _ = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [gold], files=files, extra=["--window-limit", "600"])
    assert (code, results["coverage"]["token_fit_over"]) == (0, 1)

def test_downloaded_commit_not_the_pin_exits_nonzero_naming_both_shas(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer, capsys):
    # Break: runner trusts whatever commit it got. stderr only: cloner.py:236,295 print SHA prefixes to stdout.
    served = "b" * 40
    code, _, out_dir = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [Q_ALPHA], gh=FakeGitHub(FILES, sha=served))
    err = capsys.readouterr().err
    assert (code not in (0, None), PIN[:12] in err, served[:12] in err, out_dir) == (True, True, True, [])

def test_github_archive_404_exits_nonzero_naming_repo_and_writes_no_file(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer, capsys):
    # Break: half-written baseline left in the out folder and later read as real.
    code, _, out_dir = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [Q_ALPHA], gh=FakeGitHub(FILES, archive_status=404))
    assert (code not in (0, None), "o/demo" in capsys.readouterr().err, out_dir) == (True, True, [])

def test_groq_error_row_is_marked_error_not_scored_or_refused(harness, monkeypatch, tmp_path, x1_model, x2_tokenizer):
    # Break: upstream error scored as precision 0 (mean 0.5) or counted as a refusal. Fake Groq raises APIError on every q2 call.
    groq_replies(harness, monkeypatch, fail_on=Q_BRAVO["question"])
    code, results, _ = run_eval(harness, monkeypatch, tmp_path, x2_tokenizer, [Q_ALPHA, Q_BRAVO], mode="answer")
    got = rows(results)
    assert (code, got["q1"]["status"], got["q2"]["status"], results["summary"]["mean_precision"], results["summary"]["refusals"]) == (
        0, "ok", "error", 1.0, 0)
