"""
api/routes/graph.py — GET /api/v1/graph?repo_name=fastapi

Returns the pre-built interactive dependency graph HTML + stats.

The graph is built during ingestion and cached in memory.
This route just retrieves from the cache — no recomputation.

Why not rebuild on every request?
  Building the graph requires the local repo files (to parse imports).
  We delete those after ingestion. So we cache the result at ingest time.
  Fast response, no reprocessing.
"""

from fastapi import APIRouter, HTTPException
from api.models import GraphResponse

# Import the cache from ingest.py — single source of truth
from api.routes.ingest import graph_cache
from core.graph.renderer import render_graph_html
from core.graph.serializer import deserialize_graph
from core.storage.repo_metadata import get_repo_metadata

router = APIRouter()


@router.get(
    "/graph",
    response_model=GraphResponse,
    summary="Get dependency graph",
    description="Returns interactive Pyvis HTML graph + statistics for the indexed repo.",
)
async def get_graph(repo_name: str):
    """
    Returns the dependency graph built during ingestion.
    The HTML is a self-contained Pyvis network — embed it in an iframe.
    """

    if repo_name not in graph_cache:

        metadata = get_repo_metadata(repo_name)

        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No graph found for '{repo_name}'. "
                    f"Run POST /api/v1/ingest first."
                ),
            )

        graph = deserialize_graph(
            metadata["graph_data"]
        )

        graph_cache[repo_name] = {
            "graph": graph,
            "stats": metadata["graph_stats"],
        }

        print(
            f"[graph.py] Recovered graph "
            f"for '{repo_name}' from metadata"
        )

    cached = graph_cache[repo_name]

    return GraphResponse(
        status="success",
        repo_name=repo_name,
        html=render_graph_html(cached["graph"]),
        stats=cached["stats"],
    )