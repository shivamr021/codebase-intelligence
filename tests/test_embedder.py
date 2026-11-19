"""
tests/test_embedder.py — Unit tests for embedder.py deterministic ID generation.

Tests that _chunk_id() produces stable, deterministic IDs and that
different inputs produce different IDs.
"""

from core.embeddings.embedder import _chunk_id


class TestChunkId:
    """Tests for _chunk_id() deterministic ID generation."""

    def test_deterministic(self):
        """Same input always produces the same ID."""
        id1 = _chunk_id("myrepo", "src/auth.py", "authenticate", 14)
        id2 = _chunk_id("myrepo", "src/auth.py", "authenticate", 14)
        assert id1 == id2

    def test_different_repo(self):
        """Different repo_name → different ID."""
        id1 = _chunk_id("repo_a", "src/auth.py", "authenticate", 14)
        id2 = _chunk_id("repo_b", "src/auth.py", "authenticate", 14)
        assert id1 != id2

    def test_different_file(self):
        """Different file_path → different ID."""
        id1 = _chunk_id("myrepo", "src/auth.py", "authenticate", 14)
        id2 = _chunk_id("myrepo", "src/users.py", "authenticate", 14)
        assert id1 != id2

    def test_different_name(self):
        """Different chunk_name → different ID."""
        id1 = _chunk_id("myrepo", "src/auth.py", "authenticate", 14)
        id2 = _chunk_id("myrepo", "src/auth.py", "authorize", 14)
        assert id1 != id2

    def test_different_line(self):
        """Different start_line → different ID."""
        id1 = _chunk_id("myrepo", "src/auth.py", "authenticate", 14)
        id2 = _chunk_id("myrepo", "src/auth.py", "authenticate", 50)
        assert id1 != id2

    def test_returns_hex_string(self):
        """ID should be a 32-character hex string (MD5 digest)."""
        chunk_id = _chunk_id("myrepo", "src/auth.py", "func", 1)
        assert isinstance(chunk_id, str)
        assert len(chunk_id) == 32
        # Verify it's valid hex
        int(chunk_id, 16)

    def test_no_collisions_across_similar_inputs(self):
        """Closely related inputs should not collide."""
        ids = set()
        for i in range(100):
            ids.add(_chunk_id("repo", "file.py", f"func_{i}", i))
        assert len(ids) == 100


# --- qf-chunk-id: QAD17 key (file, name, start, end, sha256(text)[:16], commit). Seams A (_build_ids_and_payloads), B (store).
import pytest
from core.embeddings.embedder import _build_ids_and_payloads
from tests.test_chunker import chunks_of

# zod packages/bench/validate-vs-safeparse.ts:141-145, pasted verbatim: outer and inner arrow are both "anonymous" at lines 1-5.
ZOD_NESTED_ARROW = """const safeParseOn = (s: z.ZodType) => () => {
  const r = s.safeParse(pool[idx++ & 63]);
  sink += r.success ? 1 : 0;
  escaped = r;
};
"""
COMMIT_A, COMMIT_B = "a" * 40, "b" * 40

def zod_pair(tmp_path):  # the 2 chunks behind the 20 baseline collisions; fixture checked, so a chunker change fails loudly here
    pair = chunks_of(tmp_path, "validate-vs-safeparse.ts", ZOD_NESTED_ARROW)
    assert [(c["file_path"], c["name"], c["start_line"], c["end_line"]) for c in pair] == [
        ("src/validate-vs-safeparse.ts", "anonymous", 1, 5)] * 2
    assert pair[0]["text"] != pair[1]["text"]
    return pair

def chunk(name="f", start=10, end=20, text="def f(): pass", file_path="src/a.py"):
    return {"file_path": file_path, "name": name, "text": text, "chunk_type": "function",
            "start_line": start, "end_line": end, "language": "python"}

def test_zod_nested_arrows_same_lines_different_text_get_two_ids(tmp_path):
    # Middle, new. Break: embedder.py:83 key omits text, so outer and inner arrow share one ID (an end-line-only fix too).
    ids, _, _ = _build_ids_and_payloads(zod_pair(tmp_path), "zod")
    assert len(set(ids)) == 2

def test_zod_nested_arrows_stored_leave_two_points_not_one(harness, tmp_path):
    # Middle, new. Break: same ID for the pair -> Qdrant upsert overwrites the outer arrow; the user loses 1 of 2 chunks.
    from core.embeddings.embedder import embed_and_store
    embed_and_store(zod_pair(tmp_path), "zod")
    assert harness.store.count("zod").count == 2

@pytest.mark.parametrize("end_a, end_b", [pytest.param(20, 21, id="off-by-one"), pytest.param(12, 57, id="far-apart")])
def test_chunks_differing_only_in_end_line_get_two_ids(end_a, end_b):
    # Middle, new. Break: embedder.py:83 key omits end_line, so a cut window and its full body share one ID.
    ids, _, _ = _build_ids_and_payloads([chunk(end=end_a), chunk(end=end_b)], "repo")
    assert len(set(ids)) == 2

def test_identical_chunk_twice_still_gets_one_id(tmp_path):
    # Edge, existing guard. Break: a random or counter part in the key -> re-index stores duplicates (QAD17 true-duplicate merge).
    outer, inner = zod_pair(tmp_path)
    ids, _, _ = _build_ids_and_payloads([outer, dict(outer), inner, dict(inner)], "zod")
    assert ids[0] == ids[1] and ids[2] == ids[3]

def test_zod_pair_at_two_commits_gets_four_ids(tmp_path):
    # Edge, new. Break: key drops text (today: 2 IDs) or drops commit_hash -> versions or arrows overwrite each other.
    pair = zod_pair(tmp_path)
    ids = _build_ids_and_payloads(pair, "zod", COMMIT_A)[0] + _build_ids_and_payloads(pair, "zod", COMMIT_B)[0]
    assert len(set(ids)) == 4

@pytest.mark.parametrize("name_a, name_b", [pytest.param("load", "save", id="sibling-names"),
                                            pytest.param("anonymous", "safeParseOn", id="arrow-vs-named")])
def test_same_lines_same_text_different_name_get_two_ids(name_a, name_b):
    # Edge, existing guard. Break: key drops chunk_name -> two named units on the same lines and text merge into one.
    ids, _, _ = _build_ids_and_payloads([chunk(name=name_a, text="x = 1"), chunk(name=name_b, text="x = 1")], "repo")
    assert len(set(ids)) == 2

def test_property_distinct_chunks_keep_distinct_ids():
    # Bad, new. Break: any key field dropped (text, end_line, name, start_line, file) -> two distinct chunks share an ID.
    # FALLBACK:TOOL hypothesis->seeded random (hypothesis not in the venv). Tiny pools make text-only differences common.
    import itertools, random
    pool = list(itertools.product(["src/a.ts", "src/b.ts"], ["anonymous", "run"], [1, 3], [5, 9],
                                  ["() => 1", "(s) => () => 1", "() => {\n  return 2;\n}", "x", "x ", "X"]))
    rng, text_only_pairs = random.Random(1119), 0
    for _ in range(300):
        picked = rng.sample(pool, rng.randint(2, 12))  # distinct (file, name, start, end, text) by construction
        text_only_pairs += len(picked) - len({p[:4] for p in picked})
        ids, _, _ = _build_ids_and_payloads([chunk(*p[1:4], text=p[4], file_path=p[0]) for p in picked], "repo")
        assert len(set(ids)) == len(ids), f"distinct chunks share an ID: {picked}"
    assert text_only_pairs > 100  # the generator really exercises same-lines-different-text
