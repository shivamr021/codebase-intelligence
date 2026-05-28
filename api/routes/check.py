"""
api/routes/check.py — Check if a repo is already indexed in Qdrant.

GET /api/v1/check?repo_name=flask-boilerplate
Returns: { "indexed": true, "chunks": 37 }

Frontend uses this to skip re-ingestion when the user types a repo
URL they've already analysed in a previous session.
"""

from fastapi import APIRouter
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from config import QDRANT_URL, QDRANT_API_KEY

router = APIRouter()

_qdrant: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant


@router.get("/check")
async def check_repo(repo_name: str):
    """Return whether a repo_name collection exists and its chunk count."""
    try:
        info = get_client().get_collection(repo_name)
        count = info.points_count or 0
        return {"indexed": count > 0, "chunks": count}
    except (UnexpectedResponse, Exception):
        return {"indexed": False, "chunks": 0}