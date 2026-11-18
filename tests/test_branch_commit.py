"""
tests/test_branch_commit.py — Tests for branch/commit-aware indexing.

Covers:
  1. Pydantic validation (IngestRequest, AskRequest)
  2. Collection naming (_build_collection_name)
  3. Commit hash in chunk IDs (_chunk_id with commit_hash)
  4. Commit hash in payloads (_build_ids_and_payloads)
  5. Mutual exclusivity of branch + commit_hash

All tests are isolated — no network calls, no Qdrant, no heavy models.
"""

import pytest
from pydantic import ValidationError

from core.embeddings.embedder import _chunk_id, _build_ids_and_payloads
from api.models import IngestRequest, AskRequest


# -----------------------------------------------------------------------
# Helper: build a minimal chunk dict for embedder tests
# -----------------------------------------------------------------------

def _make_chunk(
    file_path: str = "src/main.py",
    name: str = "main",
    start_line: int = 1,
    end_line: int = 10,
) -> dict:
    return {
        "file_path": file_path,
        "name": name,
        "text": f"def {name}(): pass",
        "chunk_type": "function",
        "start_line": start_line,
        "end_line": end_line,
        "language": "python",
    }


# -----------------------------------------------------------------------
# Test: Collection naming
# -----------------------------------------------------------------------

class TestCollectionNaming:
    """Tests for _build_collection_name() versioned collection names."""

    def setup_method(self):
        # Import here to avoid circular import issues at module load
        from api.routes.ingest import _build_collection_name
        self.build = _build_collection_name

    def test_default_no_branch_no_commit(self):
        """No branch or commit → bare repo name (backwards compatible)."""
        assert self.build("fastapi") == "fastapi"

    def test_with_branch(self):
        """Branch → repo@branch."""
        assert self.build("fastapi", branch="develop") == "fastapi@develop"

    def test_with_commit_hash(self):
        """Commit → repo@<first 8 chars>."""
        sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        assert self.build("fastapi", commit_hash=sha) == "fastapi@a1b2c3d4"

    def test_commit_takes_precedence_over_branch(self):
        """If both provided (shouldn't happen), commit wins."""
        sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        result = self.build("fastapi", branch="develop", commit_hash=sha)
        assert result == "fastapi@a1b2c3d4"

    def test_branch_with_slashes_sanitized(self):
        """Branch names with slashes get sanitized for Qdrant."""
        result = self.build("fastapi", branch="feature/auth-login")
        assert result == "fastapi@feature-auth-login"

    def test_branch_with_spaces_sanitized(self):
        """Branch names with spaces get sanitized."""
        result = self.build("fastapi", branch="my branch")
        assert result == "fastapi@my-branch"


# -----------------------------------------------------------------------
# Test: Chunk ID with commit_hash
# -----------------------------------------------------------------------

class TestChunkIdWithCommit:
    """Tests for _chunk_id() when commit_hash is provided."""

    COMMIT_A = "aaaa" * 10  # 40-char hex
    COMMIT_B = "bbbb" * 10

    def test_same_chunk_different_commits_different_ids(self):
        """Same code chunk at two different commits → different IDs."""
        id_a = _chunk_id("repo", "file.py", "func", 1, commit_hash=self.COMMIT_A)
        id_b = _chunk_id("repo", "file.py", "func", 1, commit_hash=self.COMMIT_B)
        assert id_a != id_b

    def test_same_chunk_same_commit_deterministic(self):
        """Same inputs + same commit → same ID."""
        id1 = _chunk_id("repo", "file.py", "func", 1, commit_hash=self.COMMIT_A)
        id2 = _chunk_id("repo", "file.py", "func", 1, commit_hash=self.COMMIT_A)
        assert id1 == id2

    def test_no_commit_uses_head_fallback(self):
        """No commit_hash → uses 'HEAD' as part of the hash."""
        id_none = _chunk_id("repo", "file.py", "func", 1, commit_hash=None)
        id_head = _chunk_id("repo", "file.py", "func", 1)  # default
        assert id_none == id_head  # Both should use 'HEAD'

    def test_commit_vs_no_commit_different(self):
        """Chunk with commit vs without → different IDs."""
        id_with = _chunk_id("repo", "file.py", "func", 1, commit_hash=self.COMMIT_A)
        id_without = _chunk_id("repo", "file.py", "func", 1)
        assert id_with != id_without


# -----------------------------------------------------------------------
# Test: Payloads include commit_hash
# -----------------------------------------------------------------------

class TestBuildIdsAndPayloads:
    """Tests for _build_ids_and_payloads() commit_hash injection."""

    COMMIT = "abcdef12" * 5  # 40-char hex

    def test_payload_includes_commit_hash(self):
        """When commit_hash provided, every payload should contain it."""
        chunks = [_make_chunk()]
        ids, texts, payloads = _build_ids_and_payloads(chunks, "repo", commit_hash=self.COMMIT)

        assert len(payloads) == 1
        assert payloads[0]["commit_hash"] == self.COMMIT

    def test_payload_excludes_commit_hash_when_none(self):
        """When commit_hash is None, payloads should NOT have the key."""
        chunks = [_make_chunk()]
        ids, texts, payloads = _build_ids_and_payloads(chunks, "repo", commit_hash=None)

        assert len(payloads) == 1
        assert "commit_hash" not in payloads[0]

    def test_multiple_chunks_all_have_commit(self):
        """All chunks in a batch should get the commit_hash."""
        chunks = [
            _make_chunk(name="func_a", start_line=1),
            _make_chunk(name="func_b", start_line=20),
            _make_chunk(name="func_c", start_line=40),
        ]
        ids, texts, payloads = _build_ids_and_payloads(chunks, "repo", commit_hash=self.COMMIT)

        assert len(payloads) == 3
        for p in payloads:
            assert p["commit_hash"] == self.COMMIT

    def test_ids_are_deterministic_with_commit(self):
        """IDs should be stable when commit_hash is fixed."""
        chunks = [_make_chunk()]
        ids1, _, _ = _build_ids_and_payloads(chunks, "repo", commit_hash=self.COMMIT)
        ids2, _, _ = _build_ids_and_payloads(chunks, "repo", commit_hash=self.COMMIT)
        assert ids1 == ids2

    def test_existing_payload_fields_preserved(self):
        """commit_hash should not interfere with existing payload fields."""
        chunks = [_make_chunk()]
        ids, texts, payloads = _build_ids_and_payloads(chunks, "repo", commit_hash=self.COMMIT)

        p = payloads[0]
        assert p["file_path"] == "src/main.py"
        assert p["name"] == "main"
        assert p["chunk_type"] == "function"
        assert p["language"] == "python"
        assert p["repo_name"] == "repo"
        assert p["start_line"] == 1
        assert p["end_line"] == 10


# -----------------------------------------------------------------------
# Test: Pydantic model validation
# -----------------------------------------------------------------------

class TestIngestRequestValidation:
    """Tests for IngestRequest Pydantic validation."""

    def test_valid_basic_request(self):
        """Basic request without branch/commit should pass."""
        req = IngestRequest(github_url="https://github.com/owner/repo")
        assert req.branch is None
        assert req.commit_hash is None

    def test_valid_with_branch(self):
        """Request with a valid branch name should pass."""
        req = IngestRequest(
            github_url="https://github.com/owner/repo",
            branch="develop",
        )
        assert req.branch == "develop"

    def test_valid_with_commit_hash(self):
        """Request with a valid 40-char hex commit_hash should pass."""
        sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        req = IngestRequest(
            github_url="https://github.com/owner/repo",
            commit_hash=sha,
        )
        assert req.commit_hash == sha

    def test_invalid_commit_hash_too_short(self):
        """Short commit hash should be rejected by regex."""
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(
                github_url="https://github.com/owner/repo",
                commit_hash="abc123",
            )
        assert "commit_hash" in str(exc_info.value)

    def test_invalid_commit_hash_uppercase(self):
        """Uppercase hex should be rejected (GitHub uses lowercase)."""
        with pytest.raises(ValidationError):
            IngestRequest(
                github_url="https://github.com/owner/repo",
                commit_hash="A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6A1B2",
            )

    def test_invalid_commit_hash_non_hex(self):
        """Non-hex characters should be rejected."""
        with pytest.raises(ValidationError):
            IngestRequest(
                github_url="https://github.com/owner/repo",
                commit_hash="zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
            )

    def test_branch_and_commit_mutually_exclusive(self):
        """Providing both branch AND commit_hash should raise."""
        sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(
                github_url="https://github.com/owner/repo",
                branch="develop",
                commit_hash=sha,
            )
        assert "Cannot specify both" in str(exc_info.value)

    def test_empty_branch_rejected(self):
        """Empty string branch should be rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            IngestRequest(
                github_url="https://github.com/owner/repo",
                branch="",
            )


class TestAskRequestValidation:
    """Tests for AskRequest Pydantic validation."""

    def test_valid_without_commit(self):
        """Basic request without commit_hash should pass."""
        req = AskRequest(
            repo_name="fastapi",
            question="How does auth work?",
        )
        assert req.commit_hash is None

    def test_valid_with_commit(self):
        """Request with valid commit_hash should pass."""
        sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        req = AskRequest(
            repo_name="fastapi",
            question="How does auth work?",
            commit_hash=sha,
        )
        assert req.commit_hash == sha

    def test_invalid_commit_hash(self):
        """Invalid commit_hash on AskRequest should be rejected."""
        with pytest.raises(ValidationError):
            AskRequest(
                repo_name="fastapi",
                question="How does auth work?",
                commit_hash="not-a-sha",
            )
