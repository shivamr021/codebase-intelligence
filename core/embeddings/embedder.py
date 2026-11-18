"""
core/embeddings/embedder.py — Embeds code chunks and stores them in Qdrant.

Supports two indexing modes:
  1. Full index  — first-time ingestion. Deletes stale collection, creates fresh.
  2. Incremental — re-ingestion after a commit. Upserts changed chunks, deletes
                   stale ones by file_path filter.

Key design decision: deterministic point IDs.
  Old approach: sequential ints (0, 1, 2, ...) — different on every run.
  New approach: MD5 hash of (repo_name, file_path, chunk_name, start_line).
  This means the same code chunk always gets the same ID, enabling upserts
  (Qdrant overwrites existing points with matching IDs) and targeted deletes.

Vectors come from vector_space (core/embeddings/vectors.py): it owns the model,
the vector name and the thread count. This file only stores them with upsert().
"""

import hashlib

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from config import QDRANT_URL, QDRANT_API_KEY
from core.embeddings import vectors

_qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)

UPLOAD_GROUP = 100  # points per Qdrant upsert request


def _upload(coll: str, ids: list[str], texts: list[str], payloads: list[dict]) -> None:
    """Embed and upsert in groups of UPLOAD_GROUP; create the collection when it is missing."""
    name = vectors.dense_name()
    for start in range(0, len(ids), UPLOAD_GROUP):
        end = start + UPLOAD_GROUP
        group = vectors.passage_vectors(texts[start:end])
        if start == 0 and not _qdrant.collection_exists(coll):
            _qdrant.create_collection(coll, vectors_config={
                name: VectorParams(size=len(group[0]), distance=Distance.COSINE)})
        _qdrant.upsert(coll, points=[PointStruct(id=i, vector={name: v}, payload=p)
                                     for i, v, p in zip(ids[start:end], group, payloads[start:end])])


def _chunk_id(
    repo_name: str,
    file_path: str,
    chunk_name: str,
    start_line: int,
    commit_hash: str | None = None,
) -> str:
    """
    Deterministic ID for a chunk — same input always produces the same ID.

    This is critical for incremental indexing: when a file is re-chunked,
    unchanged chunks get the same ID and are simply overwritten (no-op),
    while new/changed chunks get new IDs and are inserted.

    When commit_hash is provided, the same chunk at different commits
    gets different IDs — ensuring version isolation.

    Uses MD5 hex digest (32 chars) — Qdrant accepts string IDs.
    """
    raw = f"{repo_name}:{file_path}:{chunk_name}:{start_line}:{commit_hash or 'HEAD'}"
    return hashlib.md5(raw.encode()).hexdigest()


def _build_ids_and_payloads(
    chunks: list[dict],
    repo_name: str,
    commit_hash: str | None = None,
) -> tuple[list[str], list[str], list[dict]]:
    """
    Build deterministic IDs and payloads for a list of chunks.
    Shared by both full-index and incremental paths.

    When commit_hash is provided, it is:
      1. Included in the chunk ID hash (version isolation)
      2. Stored in every payload (queryable via filter)
    """
    ids: list[str] = []
    payloads: list[dict] = []
    texts: list[str] = []

    for chunk in chunks:
        chunk_id = _chunk_id(
            repo_name,
            chunk["file_path"],
            chunk["name"],
            chunk["start_line"],
            commit_hash,
        )
        ids.append(chunk_id)
        texts.append(chunk["text"])
        payload = {
            "text":        chunk["text"],
            "file_path":   chunk["file_path"],
            "chunk_type":  chunk["chunk_type"],
            "name":        chunk["name"],
            "start_line":  chunk["start_line"],
            "end_line":    chunk["end_line"],
            "language":    chunk["language"],
            "repo_name":   repo_name,
        }
        if commit_hash:
            payload["commit_hash"] = commit_hash
        payloads.append(payload)

    return ids, texts, payloads


def _ensure_payload_index(collection_name: str) -> None:
    """
    Create keyword indexes on file_path and commit_hash.
    - file_path index: enables fast filter-based deletes by file
    - commit_hash index: enables fast query filtering by version
    Idempotent — safe to call multiple times.
    """
    for field in ("file_path", "commit_hash"):
        try:
            _qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Index might already exist — that's fine
            pass


def _delete_if_exists(collection_name: str) -> None:
    """
    Delete a collection if it exists — clean slate for full re-indexing.
    If it doesn't exist, do nothing.

    We DON'T recreate it here — _upload() creates it with the vector name
    and size of the configured model.
    """
    try:
        existing = [c.name for c in _qdrant.get_collections().collections]
        if collection_name in existing:
            _qdrant.delete_collection(collection_name)
            print(f"[embedder.py] Deleted existing collection '{collection_name}'")
    except Exception as e:
        print(f"[embedder.py] Warning: could not check/delete collection: {e}")


def embed_and_store(
    chunks: list[dict],
    repo_name: str,
    commit_hash: str | None = None,
    collection_name: str | None = None,
) -> dict:
    """
    Full index path — embed ALL chunks and store in a fresh Qdrant collection.

    Used for first-time ingestion of a repo. Deletes any existing collection
    and rebuilds from scratch with deterministic IDs.

    collection_name defaults to repo_name for backwards compatibility.
    When indexing a branch/commit, the caller passes a versioned name.
    """
    coll = collection_name or repo_name

    if not chunks:
        return {
            "status": "error",
            "message": "No chunks provided — nothing to embed.",
            "chunks_stored": 0,
        }
    
    if len(chunks) > 1000:
        return {
            "status": "error",
            "message": (
                f"Repository too large. "
                f"{len(chunks)} chunks exceeds limit."
            ),
            "chunks_stored": 0,
        }

    # Delete stale collection — _upload() recreates it with correct params
    _delete_if_exists(coll)

    ids, texts, payloads = _build_ids_and_payloads(chunks, repo_name, commit_hash)

    try:
        print(f"[embedder.py] Embedding {len(chunks)} chunks...")

        _upload(coll, ids, texts, payloads)

        # Create payload indexes for efficient filtering and deletes
        _ensure_payload_index(coll)

        print(f"[embedder.py] Stored {len(chunks)} chunks in '{coll}'")
        return {
            "status":        "success",
            "repo_name":     repo_name,
            "chunks_stored": len(chunks),
        }

    except UnexpectedResponse as e:
        return {
            "status":  "error",
            "message": f"Qdrant rejected the upload: {str(e)}",
            "chunks_stored": 0,
        }
    except Exception as e:
        return {
            "status":  "error",
            "message": f"Embedding/storage failed: {str(e)}",
            "chunks_stored": 0,
        }


def upsert_chunks(
    chunks: list[dict],
    repo_name: str,
    commit_hash: str | None = None,
    collection_name: str | None = None,
) -> dict:
    """
    Incremental path — embed and upsert only the given chunks.

    Used when re-indexing a repo that has changed. Only the added/modified
    files are chunked and passed here. Qdrant treats matching IDs as
    overwrites (upserts).
    """
    coll = collection_name or repo_name

    if not chunks:
        return {
            "status": "success",
            "repo_name": repo_name,
            "chunks_stored": 0,
        }

    ids, texts, payloads = _build_ids_and_payloads(chunks, repo_name, commit_hash)

    try:
        print(f"[embedder.py] Upserting {len(chunks)} chunks into '{coll}'...")

        _upload(coll, ids, texts, payloads)

        print(f"[embedder.py] Upserted {len(chunks)} chunks in '{coll}'")
        return {
            "status":        "success",
            "repo_name":     repo_name,
            "chunks_stored": len(chunks),
        }

    except UnexpectedResponse as e:
        return {
            "status":  "error",
            "message": f"Qdrant rejected the upsert: {str(e)}",
            "chunks_stored": 0,
        }
    except Exception as e:
        return {
            "status":  "error",
            "message": f"Upsert failed: {str(e)}",
            "chunks_stored": 0,
        }


def delete_file_chunks(repo_name: str, file_paths: list[str]) -> int:
    """
    Delete all Qdrant points belonging to the given file paths.

    Used during incremental indexing to remove chunks from:
      - Deleted files (file no longer exists in the repo)
      - Modified files (old chunks removed before upserting new ones,
        since a modified file may produce different chunks)

    Uses a payload filter on file_path, which is indexed as a keyword
    for fast lookups (see _ensure_payload_index).
    """
    if not file_paths:
        return 0

    try:
        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchAny,
        )

        _qdrant.delete(
            collection_name=repo_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="file_path",
                        match=MatchAny(any=file_paths),
                    )
                ]
            ),
        )

        print(
            f"[embedder.py] Deleted chunks for {len(file_paths)} files "
            f"from '{repo_name}'"
        )
        return len(file_paths)

    except Exception as e:
        print(f"[embedder.py] Failed to delete file chunks: {e}")
        return 0