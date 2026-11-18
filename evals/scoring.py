"""Eval scoring (P1): line-overlap hits, Recall@5, MRR, flip table, sign test, citations, coverage.

Span = dict with "file_path", "start_line", "end_line" (extra keys such as "name" are ignored).
Rules follow PROOF-blueprint.md AD2 (metrics), AD2b (coverage) and AD3 (every gold row has a range).
"""
import json
import math
import re

from core.llm.prompts import ANSWER_QUESTION

# The one refusal sentence the answer prompt asks for, read from the prompt so the two never drift.
REFUSAL = re.search(r'say "([^"]+)"', ANSWER_QUESTION).group(1)

_CITE = re.compile(
    r"(?P<file>\b[\w./-]*\w\.[A-Za-z]\w{0,4})(?::(?P<a>\d+)(?:-(?P<b>\d+))?)?"
    r"|\blines?\s+(?P<la>\d+)(?:\s*-\s*(?P<lb>\d+))?",
    re.IGNORECASE,
)


def _overlaps(a, b):
    return a["file_path"] == b["file_path"] and a["start_line"] <= b["end_line"] and b["start_line"] <= a["end_line"]


def _inside(a, b):
    return a["file_path"] == b["file_path"] and b["start_line"] <= a["start_line"] and a["end_line"] <= b["end_line"]


def is_hit(result, gold):
    """True when result and gold share the full file_path and at least 1 line (AD2); names never count."""
    return _overlaps(result, gold)


def score_question(results, golds):
    """
    Ranked results vs golds -> {"hit@5", "hit@20", "rank", "rr", "recall@5", "all_golds@5"}.
    rank: 1-based first result hitting any gold, or None; rr = 1/rank, no cut-off, 0.0 on none;
    recall@5: share of golds hit by some top-5 result.
    """
    rank = next((i for i, r in enumerate(results, 1) if any(is_hit(r, g) for g in golds)), None)
    found = sum(any(is_hit(r, g) for r in results[:5]) for g in golds)
    return {
        "hit@5": rank is not None and rank <= 5,
        "hit@20": rank is not None and rank <= 20,
        "rank": rank,
        "rr": 1.0 / rank if rank else 0.0,
        "recall@5": found / len(golds) if golds else 0.0,
        "all_golds@5": bool(golds) and found == len(golds),
    }


def flip_table(before, after):
    """
    {qid: rank or None} x2 -> {"b", "c", "rows": {qid: {...}}}. Hit = rank <= 5.
    b = hit->miss, c = miss->hit; rows keyed by id; both runs must hold the same ids.
    """
    if set(before) != set(after):
        raise ValueError(f"runs hold different question ids: {sorted(set(before) ^ set(after))}")
    rows, b, c = {}, 0, 0
    for qid in before:
        hb = before[qid] is not None and before[qid] <= 5
        ha = after[qid] is not None and after[qid] <= 5
        b += hb and not ha
        c += ha and not hb
        rows[qid] = {"rank_before": before[qid], "rank_after": after[qid], "hit_before": hb, "hit_after": ha}
    return {"b": b, "c": c, "rows": rows}


def sign_test(b, c):
    """Exact two-sided sign test -> (p, gain); p = 1.0 at b + c = 0; gain only when p < 0.05 and c > b."""
    n = b + c
    if n == 0:
        return 1.0, False
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return p, p < 0.05 and c > b


def load_gold(path):
    """JSON rows (AD3); ValueError naming the id on a repeated id, no golds, or a gold without an int range."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    seen = set()
    for row in rows:
        qid = row.get("id")
        if qid in seen:
            raise ValueError(f"gold row id {qid!r} appears twice")
        seen.add(qid)
        if not row.get("golds"):
            raise ValueError(f"gold row {qid!r} has no golds")
        for g in row["golds"]:
            s, e = g.get("start_line"), g.get("end_line")
            if not g.get("file_path") or not isinstance(s, int) or not isinstance(e, int) or s > e:
                raise ValueError(f"gold row {qid!r} has a gold without file_path and line range: {g}")
    return rows


def parse_citations(answer):
    """`file:N`, `file:N-M`, `line N`, `lines N-M` -> [(file, start, end)]; bare lines bind to the last file named."""
    cites, last_file = [], None
    for m in _CITE.finditer(answer):
        if m.group("file"):
            last_file = m.group("file")
            if m.group("a"):
                a = int(m.group("a"))
                cites.append((last_file, a, int(m.group("b") or a)))
        elif last_file:
            a = int(m.group("la"))
            cites.append((last_file, a, int(m.group("lb") or a)))
    return cites


def score_answer(answer, golds, shown):
    """refused only for the exact refusal sentence (ruling 5); precision vs gold lines; shown_only vs shown."""
    if answer.strip() == REFUSAL:
        return {"refused": True, "precision": None, "citations": []}
    citations, correct = [], 0
    for cite in parse_citations(answer):
        c = {"file_path": cite[0], "start_line": cite[1], "end_line": cite[2]}
        correct += any(_overlaps(c, g) for g in golds)
        citations.append({
            "cite": cite,
            "strict_inside": any(_inside(c, g) for g in golds),
            "shown_only": any(_inside(c, s) for s in shown),
        })
    return {"refused": False, "precision": correct / len(citations) if citations else 0.0, "citations": citations}


def score_answers(items):
    """Refusals counted, not scored; mean_precision over scored answers (None when none)."""
    scored = [score_answer(*item) for item in items]
    kept = [s["precision"] for s in scored if not s["refused"]]
    return {"mean_precision": sum(kept) / len(kept) if kept else None, "refusals": len(scored) - len(kept)}


def missing_lines(units, chunks, sources):
    """Line covered only if inside a same-file chunk range AND the chunk text holds it at its position (stripped)."""
    missing = set()
    for unit in units:
        src = sources[unit["file_path"]].split("\n")
        for line in range(unit["start_line"], unit["end_line"] + 1):
            want = src[line - 1].strip() if line <= len(src) else None
            covered = any(
                ch["file_path"] == unit["file_path"] and ch["start_line"] <= line <= ch["end_line"]
                and want is not None
                and line - ch["start_line"] < len(ch.get("text", "").split("\n"))
                and ch["text"].split("\n")[line - ch["start_line"]].strip() == want
                for ch in chunks
            )
            if not covered:
                missing.add((unit["file_path"], line))
    return sorted(missing)
