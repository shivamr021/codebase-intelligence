"""evals/run_eval.py — eval runner (P2; PROOF-blueprint AD1, AD2, AD2b).

Pinned download -> walk_repo -> chunk_file -> embed_and_store into a fresh in-memory Qdrant (the app's own
path, api/routes/ingest.py:382-560) -> per-question scores, stats, coverage, premise pilots. Usage:
    python -m evals.run_eval --repo URL --pin SHA --gold PATH.jsonl --mode retrieval|answer --out PATH.json
                             --model-dir DIR [--window-limit N] [--pilot]
Exit 2, reason on stderr, no results file: GitHub serves another commit than the pin, or the download fails.
"""
import argparse, contextlib, hashlib, json, os, resource, shutil, sys, tempfile, time  # noqa: E401
from pathlib import Path

from qdrant_client import QdrantClient
from tokenizers import Tokenizer

from core.embeddings import embedder
from core.ingestion import chunker, cloner, walker
from core.llm import client
from core.llm.prompts import ANSWER_QUESTION, format_chunks_for_prompt
from core.retrieval import retriever
from evals.scoring import is_hit, load_gold, missing_lines, score_answers, score_question

TOP_K = 5             # production top_k [code query.py:45]
WINDOW_LIMIT = 512    # the model reads min(model_max_length, 512) tokens [code fastembed common/models.py:8,35]
ANSWER_REPEATS = 3    # [doc _maps/measure.md:182]; temperature 0.3 as /ask [code query.py:69]


def _args(argv):
    p = argparse.ArgumentParser(prog="python -m evals.run_eval")
    for name in ("--repo", "--pin", "--gold", "--out", "--model-dir"):
        p.add_argument(name, required=True)
    p.add_argument("--mode", choices=("retrieval", "answer"), required=True)
    p.add_argument("--window-limit", type=int, default=WINDOW_LIMIT)
    p.add_argument("--pilot", action="store_true", help="also score a known-bad index (every chunk cut to its first line)")
    return p.parse_args(argv)


def _load_gold(path):
    """JSONL gold rows, checked by scoring.load_gold (which reads a JSON array)."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(rows, tmp)
    try:
        return load_gold(tmp.name)
    finally:
        os.unlink(tmp.name)


def _fetch(url, pin):
    """Download the repo at the pin -> (owner/repo, full sha, local_path, work_dir); RuntimeError on any doubt."""
    try:
        r = cloner.resolve_repo(url, commit_hash=pin)
    except ValueError as e:
        raise RuntimeError(f"{url}: {e}") from e
    if r["status"] == "error":
        raise RuntimeError(r["message"])
    name, sha = f"{r['owner']}/{r['repo']}", r["commit_sha"]
    if not sha.lower().startswith(pin.lower()):
        raise RuntimeError(f"{name}: GitHub serves commit {sha[:12]}, not the pin {pin[:12]}")
    c = cloner.clone_repo(url, r["owner"], r["repo"], commit_hash=sha)
    if c["status"] == "error":
        raise RuntimeError(f"{name}: download at pin {pin[:12]} failed: {c['message']}")
    if sha not in os.path.basename(c["local_path"]):  # GitHub zips hold one folder <repo>-<sha>/
        shutil.rmtree(c["work_dir"], ignore_errors=True)
        raise RuntimeError(f"{name}: archive folder {os.path.basename(c['local_path'])} is not the pin {pin[:12]}")
    return name, sha, c["local_path"], c["work_dir"]


@contextlib.contextmanager
def _patched(module, **values):
    """Set module globals for the block (chunker limits, Qdrant clients); the old values come back after."""
    old = {k: getattr(module, k) for k in values}
    vars(module).update(values)
    try:
        yield
    finally:
        vars(module).update(old)


def _dropped(fn, *args):
    """Pieces `fn` makes only when MIN_CHUNK_CHARS is 0, i.e. the ones the minimum-size rule drops (AD5)."""
    kept = {(c["start_line"], c["end_line"], c["name"]) for c in fn(*args)}
    with _patched(chunker, MIN_CHUNK_CHARS=0):
        return [c for c in fn(*args) if (c["start_line"], c["end_line"], c["name"]) not in kept]


def _chunk(files):
    """chunk_file per walked file, as ingest does; plus uncut units (AD2b) and min-size drops, same file reads."""
    chunks, units, dropped, sources = [], [], [], {}
    for f in files:
        with open(f["path"], encoding="utf-8", errors="replace") as fh:
            sources[f["rel_path"]] = src = fh.read()
        chunks += chunker.chunk_file(f)
        with _patched(chunker, MIN_CHUNK_CHARS=0, MAX_CHUNK_CHARS=10**9):
            units += chunker.chunk_file(f)
        args = (src, f["language"], f["rel_path"])
        ast = chunker._chunk_by_ast(*args)
        dropped += _dropped(chunker._chunk_by_ast, *args)
        dropped += _dropped(chunker._module_chunks, *args, ast) if ast else _dropped(chunker._fallback_line_chunks, *args)
    return chunks, units, dropped, sources


def _store(chunks, coll, sha):
    out = embedder.embed_and_store(chunks, coll, commit_hash=sha, collection_name=coll)
    if out["status"] == "error":
        raise RuntimeError(f"embedding failed: {out['message']}")


def _retrieval_rows(gold, coll, stored):
    """Full ranking (no cut-off) per question; indexed = some stored chunk overlaps a gold (AD2 classes)."""
    rows = []
    for q in gold:
        found = retriever.retrieve_chunks(q["question"], coll, top_k=len(stored), collection_name=coll)
        s = score_question(found, q["golds"])
        indexed = any(is_hit(c, g) for c in stored for g in q["golds"])
        rows.append({"id": q["id"], "class": "HIT" if s["hit@5"] else "OUTRANKED" if indexed else "NOT_INDEXED", **s})
    return rows


def _retrieval_summary(rows):
    n = len(rows) or 1
    return {"recall@5": round(sum(r["recall@5"] for r in rows) / n, 4), "mrr": round(sum(r["rr"] for r in rows) / n, 4),
            "hit@5": sum(r["hit@5"] for r in rows), "hit@20": sum(r["hit@20"] for r in rows),
            "classes": {c: sum(r["class"] == c for r in rows) for c in ("HIT", "OUTRANKED", "NOT_INDEXED")}}


def _answer_rows(gold, coll):
    """The /ask prompt (query.py:42-69) asked ANSWER_REPEATS times; an upstream error marks the row, never scores it."""
    rows, scored = [], []
    for q in gold:
        shown = retriever.retrieve_chunks(q["question"], coll, top_k=TOP_K, collection_name=coll)
        prompt = ANSWER_QUESTION.format(code_context=format_chunks_for_prompt(shown), question=q["question"])
        answers = [client.call_llm(prompt, temperature=0.3) for _ in range(ANSWER_REPEATS)]
        if any(a.startswith("[LLM Error]") for a in answers):
            rows.append({"id": q["id"], "status": "error", "precision": None, "answers": answers})
            continue
        items = [(a, q["golds"], shown) for a in answers]
        scored += items
        s = score_answers(items)
        rows.append({"id": q["id"], "status": "ok", "precision": s["mean_precision"], "refusals": s["refusals"], "answers": answers})
    return rows, score_answers(scored)


def _token_fit_over(chunks, model_dir, limit):
    """Stored text counted by the model's own tokenizer.json, truncation off, 2 special tokens included (ruling 3)."""
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    return sum(len(tok.encode(c["text"]).ids) > limit for c in chunks)


def _gold_lines_in(pieces, gold):
    lines = {(g["file_path"], n) for q in gold for g in q["golds"] for n in range(g["start_line"], g["end_line"] + 1)}
    return sum(any(p["file_path"] == f and p["start_line"] <= n <= p["end_line"] for p in pieces) for f, n in lines)


def main(argv: list[str] | None = None) -> int:
    a = _args(argv)
    t0 = time.time()
    gold = _load_gold(a.gold)
    try:
        name, sha, local_path, work_dir = _fetch(a.repo, a.pin)
    except RuntimeError as e:
        print(f"[run_eval] {e}", file=sys.stderr)
        return 2
    try:
        files = walker.walk_repo(local_path)
        chunks, units, dropped, sources = _chunk(files)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    coll = "eval--" + name.replace("/", "--").lower()
    res = {"repo": name, "pin": a.pin, "commit": sha, "mode": a.mode, "collection": coll}
    store = QdrantClient(":memory:")  # AD1: embedder and retriever share one fresh in-memory store, never the app's
    with _patched(embedder, _qdrant=store), _patched(retriever, _qdrant=store):
        _store(chunks, coll, sha)
        rss_after = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 1024
        if a.mode == "retrieval":
            res["questions"] = _retrieval_rows(gold, coll, chunks)
            res["summary"] = _retrieval_summary(res["questions"])
        else:
            res["questions"], res["summary"] = _answer_rows(gold, coll)
        if a.pilot:
            bad = [{**c, "text": c["text"].split("\n")[0], "end_line": c["start_line"]} for c in chunks]
            _store(bad, coll + "--known-bad", sha)
            recall = {k: _retrieval_summary(_retrieval_rows(gold, n, s))["recall@5"]
                      for k, n, s in (("real", coll, chunks), ("known_bad", coll + "--known-bad", bad))}
            res["pilot"] = {"recall@5": recall, "missing_lines": {"real": len(missing_lines(units, chunks, sources)),
                                                                  "known_bad": len(missing_lines(units, bad, sources))}}
    res["coverage"] = {"token_fit_over": _token_fit_over(chunks, a.model_dir, a.window_limit), "window_limit": a.window_limit,
                       "missing_lines": len(missing_lines(units, chunks, sources))}
    res["min_chunk_drop"] = {"dropped": len(dropped), "gold_lines_in_dropped": _gold_lines_in(dropped, gold)}
    res["stats"] = {"files": len(files), "chunks": len(chunks), "model_sha256": hashlib.sha256(Path(a.model_dir, "model_optimized.onnx").read_bytes()).hexdigest(),
                    "seconds": round(time.time() - t0, 2), "rss_after_embed_kb": rss_after,
                    "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    text = json.dumps(res, indent=2)  # serialise first: a failure leaves no half-written file
    with open(a.out + ".tmp", "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(a.out + ".tmp", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
