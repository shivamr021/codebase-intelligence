"""
core/embeddings/embedder.py — Embeds code chunks and stores them in Qdrant.

Key fix from original version:
  We removed _ensure_collection() which manually created collections with
  explicit VectorParams. This conflicted with _qdrant.add() which uses
  FastEmbed and creates its own collection format internally.

  Correct approach:
    1. Delete existing collection if present (clean slate)
    2. Call _qdrant.add() directly — it creates the collection automatically
       with the correct vector config for the FastEmbed model being used.

  Never manually create a collection AND use _qdrant.add() together.
  Pick one. We pick _qdrant.add() because it handles everything.
"""

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from config import QDRANT_URL, QDRANT_API_KEY

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)


def _delete_if_exists(collection_name: str) -> None:
    """
    Delete a collection if it exists — clean slate for re-indexing.
    If it doesn't exist, do nothing.

    We DON'T recreate it here — _qdrant.add() creates it automatically
    with the correct vector params for the FastEmbed model.
    """
    try:
        existing = [c.name for c in _qdrant.get_collections().collections]
        if collection_name in existing:
            _qdrant.delete_collection(collection_name)
            print(f"[embedder.py] Deleted existing collection '{collection_name}'")
    except Exception as e:
        print(f"[embedder.py] Warning: could not check/delete collection: {e}")


def embed_and_store(chunks: list[dict], repo_name: str) -> dict:
    """
    Embed code chunks using FastEmbed and store in Qdrant.

    _qdrant.add() does three things internally:
      1. Embeds each document using BAAI/bge-small-en-v1.5 (FastEmbed)
      2. Creates the Qdrant collection if it doesn't exist
      3. Upserts all points (vector + payload) into the collection

    We only need to ensure no stale collection exists before calling it.
    """

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

    # Delete stale collection — let _qdrant.add() recreate with correct params
    _delete_if_exists(repo_name)

    texts = [chunk["text"] for chunk in chunks]

    payloads = [
        {
            "text":       chunk["text"],
            "file_path":  chunk["file_path"],
            "chunk_type": chunk["chunk_type"],
            "name":       chunk["name"],
            "start_line": chunk["start_line"],
            "end_line":   chunk["end_line"],
            "language":   chunk["language"],
            "repo_name":  repo_name,
        }
        for chunk in chunks
    ]

    ids = list(range(len(chunks)))

    try:
        print(f"[embedder.py] Embedding {len(chunks)} chunks (first run downloads ~130MB model)...")

        _qdrant.add(
            collection_name=repo_name,
            documents=texts,
            metadata=payloads,
            ids=ids,
            batch_size=100,
        )

        print(f"[embedder.py] Stored {len(chunks)} chunks in '{repo_name}'")
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