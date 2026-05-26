"""
core/embeddings/embedder.py — Embeds code chunks and stores them in Qdrant.

What this file does:
  Takes the list of chunks from chunker.py, converts each chunk's text
  into a vector (embedding), and stores that vector + metadata in Qdrant.

What is an embedding?
  An embedding is a list of numbers (a vector) that represents the
  *meaning* of a piece of text in mathematical space.

  Example: "authenticate user with password" and "login function with credentials"
  will have vectors that are numerically close to each other — even though
  they share no words. This is how semantic search works.

  We use FastEmbed's BAAI/bge-small-en-v1.5 model:
    - 384-dimensional vectors (384 numbers per chunk)
    - ~130MB model download on first run (cached after that)
    - Runs on CPU — no GPU needed
    - Fast enough for repos up to 1000 chunks in under 2 minutes

Why FastEmbed instead of Gemini Embedding API?
  Earlier we planned to use Gemini Embedding API. We switched because:
  1. Gemini quota issues (as you saw with the 429 error)
  2. FastEmbed runs locally — zero API dependency, zero quota risk
  3. Qdrant's Python client has FastEmbed built in — no extra package
  4. Same embedding model is used for both storage AND retrieval,
     which is required for cosine similarity to work correctly.

How Qdrant stores data:
  Qdrant organises vectors into "collections" (like tables in a database).
  Each collection has "points" — each point is:
    - id:      a unique integer
    - vector:  the embedding (list of 384 floats)
    - payload: metadata dict (file_path, name, chunk_type, etc.)

  We create one collection per repo (named after the repo).
  This lets us query "find relevant chunks from THIS repo" specifically.

Interview talking point:
  "I use cosine similarity for retrieval. Cosine similarity measures the
  angle between two vectors — a score of 1.0 means identical meaning,
  0.0 means unrelated. It works better than Euclidean distance for text
  embeddings because it's scale-invariant — a long function and a short
  function about the same concept will still score similarly."
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from config import QDRANT_URL, QDRANT_API_KEY

# -----------------------------------------------------------------------
# FastEmbed model configuration.
#
# BAAI/bge-small-en-v1.5 is the model name on HuggingFace.
# Qdrant's client downloads it automatically on first use.
# 384 is the vector dimension this model produces — must match
# what we tell Qdrant when creating the collection.
#
# Do NOT change EMBEDDING_MODEL without also changing VECTOR_SIZE.
# Mismatched dimensions = Qdrant rejects every insert with an error.
# -----------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384

# -----------------------------------------------------------------------
# Qdrant client — initialised once at module level.
#
# url + api_key authenticates with your Qdrant Cloud cluster.
# timeout=60 — some operations (creating collection, bulk insert) can
# take several seconds on the free tier. 60s prevents false timeouts.
# -----------------------------------------------------------------------
_qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)


def _ensure_collection(collection_name: str) -> None:
    """
    Creates a Qdrant collection for this repo if it doesn't exist.
    If it already exists (re-indexing the same repo), deletes and recreates it.

    Why delete and recreate?
      If the user re-submits the same repo URL, we want fresh embeddings.
      Stale vectors from an old version of the code would corrupt results.
      Fresh collection = guaranteed consistency.

    VectorParams:
      size     = number of dimensions (must match VECTOR_SIZE)
      distance = similarity metric (Cosine for text embeddings)
    """
    # Check if collection already exists
    existing = [c.name for c in _qdrant.get_collections().collections]

    if collection_name in existing:
        # Delete old collection — clean slate for re-indexing
        print(f"[embedder.py] Collection '{collection_name}' exists — recreating.")
        _qdrant.delete_collection(collection_name)

    # Create fresh collection
    _qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"[embedder.py] Created collection '{collection_name}'")


def embed_and_store(chunks: list[dict], repo_name: str) -> dict:
    """
    Public interface — called from the ingestion pipeline.

    Takes:
      chunks    : list of chunk dicts from chunker.py
      repo_name : string used as the Qdrant collection name

    Returns:
      {
        "status":        "success",
        "repo_name":     "fastapi",
        "chunks_stored": 142,
      }

    Flow:
      1. Create/recreate Qdrant collection
      2. Extract text from chunks for embedding
      3. Generate embeddings using FastEmbed (built into qdrant_client)
      4. Build Qdrant PointStruct objects (id + vector + payload)
      5. Upload in batches of 100
    """

    if not chunks:
        return {
            "status": "error",
            "message": "No chunks provided — nothing to embed.",
            "chunks_stored": 0,
        }

    # --- Step 1: Prepare collection ---
    try:
        _ensure_collection(repo_name)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create Qdrant collection: {str(e)}",
            "chunks_stored": 0,
        }

    # --- Step 2: Extract texts for embedding ---
    # We embed the chunk text, but we store the full metadata as payload.
    texts = [chunk["text"] for chunk in chunks]

    # --- Step 3 + 4: Embed and build points ---
    # _qdrant.add() is the FastEmbed-integrated method.
    # It handles embedding internally — we just pass texts + payloads.
    # This is cleaner than calling a separate embedding model ourselves.
    #
    # batch_size=100: embed 100 chunks at a time.
    # Larger batches = faster but more RAM. 100 is safe for Railway's limits.
    try:
        # Build payload list — one dict per chunk
        # This is what gets stored alongside the vector in Qdrant.
        # Everything here is retrievable at query time.
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

        # Generate IDs — Qdrant requires integer or UUID point IDs.
        # We use simple sequential integers starting from 0.
        ids = list(range(len(chunks)))

        # _qdrant.add() with encoder="fastembed" uses FastEmbed locally.
        # It downloads the model on first call (~130MB, cached after that).
        _qdrant.add(
            collection_name=repo_name,
            documents=texts,         # FastEmbed embeds these
            metadata=payloads,       # stored as point payload
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