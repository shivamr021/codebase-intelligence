"""
core/retrieval/retriever.py — Semantic search over indexed code chunks.

What this file does:
  When a user asks a question ("where is authentication handled?"),
  this file:
    1. Embeds the question using the SAME model used during indexing
    2. Searches Qdrant for the top-k most semantically similar chunks
    3. Returns those chunks as context for the LLM prompt

Why must we use the SAME embedding model for queries and storage?
  Cosine similarity compares two vectors in the same mathematical space.
  If you embed chunks with model A and queries with model B, the vectors
  live in different spaces — similarity scores are meaningless garbage.
  Same model = same space = meaningful similarity scores.

  This is one of the most common RAG bugs in student projects —
  knowing this makes you look sharp in interviews.

What is top-k retrieval?
  We don't give the LLM ALL chunks (that would overflow the context window
  and cost too many tokens). We retrieve only the top-k most relevant ones.
  k=5 is our default — enough context without overwhelming the LLM.

  The tradeoff: higher k = more context = potentially better answers,
  but also more tokens used per request = faster quota exhaustion on Groq.
"""

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from config import QDRANT_URL, QDRANT_API_KEY

# Reuse the same client configuration as embedder.py
# Same model name is critical — see module docstring above
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)

# Default number of chunks to retrieve per query.
# 5 chunks × ~750 tokens each = ~3750 tokens of context.
# Well within Groq's 6000 TPM limit per request.
DEFAULT_TOP_K = 5


def retrieve_chunks(
    query: str,
    repo_name: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Embed a query and return the top-k most relevant chunks from Qdrant.

    Args:
        query     : Natural language question or search term
        repo_name : Qdrant collection to search in (= repo name from cloner)
        top_k     : Number of chunks to return

    Returns:
        List of chunk dicts, each containing:
        {
            "text":       "def authenticate(user, pwd): ...",
            "file_path":  "src/auth.py",
            "chunk_type": "function",
            "name":       "authenticate",
            "start_line": 14,
            "end_line":   38,
            "language":   "python",
            "score":      0.87,   ← cosine similarity (0.0 to 1.0)
        }
        Returns empty list on any error — never raises.

    How _qdrant.query() works:
        It embeds the query text using FastEmbed (same model as indexing),
        then searches the collection for the nearest vectors using
        cosine similarity. Returns ScoredPoint objects with .payload and .score.
    """

    # First check the collection exists — gives a clearer error than
    # the cryptic Qdrant "collection not found" exception
    try:
        existing = [c.name for c in _qdrant.get_collections().collections]
        if repo_name not in existing:
            print(
                f"[retriever.py] Collection '{repo_name}' not found. "
                f"Has this repo been ingested yet?"
            )
            return []
    except Exception as e:
        print(f"[retriever.py] Cannot connect to Qdrant: {e}")
        return []

    # Perform semantic search
    try:
        # _qdrant.query() is the FastEmbed-integrated search method.
        # It handles embedding the query internally.
        # query_text  → gets embedded → vector search → top_k results
        results = _qdrant.query(
            collection_name=repo_name,
            query_text=query,
            limit=top_k,
        )

        # results is a list of QueryResponse objects.
        # Each has: .id, .score, .metadata (our payload)
        chunks = []
        for result in results:
            # result.metadata contains everything we stored in the payload
            chunk = dict(result.metadata)

            # Add the similarity score — useful for debugging and
            # for filtering low-quality results
            chunk["score"] = round(result.score, 4)
            chunks.append(chunk)

        print(
            f"[retriever.py] Query: '{query[:50]}...' → "
            f"{len(chunks)} chunks retrieved from '{repo_name}'"
        )
        return chunks

    except UnexpectedResponse as e:
        print(f"[retriever.py] Qdrant search error: {e}")
        return []
    except Exception as e:
        print(f"[retriever.py] Unexpected retrieval error: {e}")
        return []


def retrieve_all_chunks(repo_name: str, limit: int = 50) -> list[dict]:
    """
    Retrieve up to `limit` chunks from a collection without a query.
    Used for architecture analysis and bug detection — where we want
    a broad sample of the codebase rather than query-specific results.

    Qdrant's scroll() method pages through all points in a collection.
    We use it here to get a representative sample of the codebase.
    """
    try:
        # scroll() returns (list_of_records, next_page_offset)
        # We only need the first page for our use case
        records, _ = _qdrant.scroll(
            collection_name=repo_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,  # Don't return vectors — saves bandwidth
        )

        chunks = []
        for record in records:
            chunk = dict(record.payload)
            chunk["score"] = 1.0  # No relevance score for scroll — set to 1.0
            chunks.append(chunk)

        print(
            f"[retriever.py] Scrolled {len(chunks)} chunks from '{repo_name}'"
        )
        return chunks

    except Exception as e:
        print(f"[retriever.py] Scroll error for '{repo_name}': {e}")
        return []