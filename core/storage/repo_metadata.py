from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
import hashlib

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_METADATA_COLLECTION,
)

# Separate client so this module is self-contained
_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def ensure_metadata_collection():
    """
    Creates repo_metadata collection if it does not exist.

    We store metadata only, but Qdrant still requires a vector schema.
    We'll use a tiny dummy vector.
    """

    collections = _client.get_collections().collections

    existing = {c.name for c in collections}

    if QDRANT_METADATA_COLLECTION in existing:
        return

    from qdrant_client.http.models import PayloadSchemaType

    _client.create_collection(
        collection_name=QDRANT_METADATA_COLLECTION,
        vectors_config=VectorParams(
            size=1,
            distance=Distance.COSINE,
        ),
    )

    _client.create_payload_index(
        collection_name=QDRANT_METADATA_COLLECTION,
        field_name="repo_name",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    print(
        f"[repo_metadata.py] Created collection "
        f"'{QDRANT_METADATA_COLLECTION}'"
    )


def save_repo_metadata(
    repo_name,
    summary,
    mermaid,
    graph_stats,
    graph_data,
    architecture_context=None,
    commit_sha=None,
    file_hashes=None,
):
    """
    Store architecture + graph artifacts for a repository.
    Existing repo metadata is overwritten.

    commit_sha:  The HEAD commit SHA at index time (for incremental indexing).
    file_hashes: Dict of {rel_path: sha256_hex} for every indexed file.
    """

    ensure_metadata_collection()

    # Remove old record if it exists
    metadata_id = hashlib.md5(
        repo_name.encode()
    ).hexdigest()

    point = PointStruct(
        id=metadata_id,
        vector=[0.0],
        payload={
            "repo_name": repo_name,
            "summary": summary,
            "mermaid": mermaid,
            "graph_stats": graph_stats,
            "graph_data": graph_data,
            "architecture_context": architecture_context,
            "commit_sha": commit_sha,
            "file_hashes": file_hashes or {},
        },
    )

    _client.upsert(
        collection_name=QDRANT_METADATA_COLLECTION,
        wait=True,
        points=[point],
    )

    print(
        f"[repo_metadata.py] Stored metadata for '{repo_name}'"
        f" (commit: {commit_sha[:8] + '...' if commit_sha else 'N/A'})"
    )


def get_repo_metadata(repo_name: str):
    """
    Retrieve persisted metadata for a repository.
    Returns None when nothing is stored yet, including on a fresh Qdrant
    where the metadata collection does not exist.
    """

    if not _client.collection_exists(QDRANT_METADATA_COLLECTION):
        return None

    results, _ = _client.scroll(
        collection_name=QDRANT_METADATA_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="repo_name",
                    match=MatchValue(value=repo_name),
                )
            ]
        ),
        limit=1,
    )

    if not results:
        return None

    return results[0].payload