"""
api/routes/ingest.py — POST /api/v1/ingest

This is the entry point for the entire system.
One request triggers: clone → walk → chunk → embed → graph → store.

Flow:
  1. Validate GitHub URL (cloner does this)
  2. Clone the repo to /tmp/
  3. Walk file tree → list of source files
  4. Chunk each file via AST → list of code chunks
  5. Embed chunks + store in Qdrant
  6. Build dependency graph + store in memory cache
  7. Clean up cloned repo from disk
  8. Return summary stats

Why clean up the clone in step 7?
  Railway's disk is limited. Keeping clones around wastes space.
  All the important data (chunks, embeddings) is now in Qdrant Cloud.
  The graph is in the in-memory cache (see graph_cache below).
  We don't need the raw files anymore.

Graph cache:
  The dependency graph (NetworkX DiGraph) is stored in a module-level
  dict keyed by repo_name. This is an in-memory cache.
  Limitation: if Railway restarts, the cache is empty — user must re-ingest.
  For a demo project this is acceptable. In production you'd serialise
  the graph to a database.

Interview talking point:
  "The ingestion pipeline is sequential by design — each step's output
  is the next step's input. This makes it easy to debug: if something
  breaks, the print logs tell you exactly which step failed and why.
  I clean up the local clone after chunking because all the data I need
  is persisted in Qdrant Cloud — the raw files are just temporary."
"""

import shutil
import stat
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

def _force_remove_readonly(func, path, _):
    """Windows read-only file handler for shutil.rmtree — see cloner.py for explanation."""
    import os
    os.chmod(path, stat.S_IWRITE)
    func(path)


from api.models import IngestRequest, IngestResponse, ErrorResponse
from core.ingestion.cloner import clone_repo
from core.ingestion.walker import walk_repo
from core.ingestion.chunker import chunk_files
from core.embeddings.embedder import embed_and_store
from core.graph.builder import build_dependency_graph, get_graph_stats
from core.graph.renderer import render_graph_html

router = APIRouter()

# -----------------------------------------------------------------------
# In-memory graph cache.
# Key:   repo_name (str)
# Value: dict with "graph" (nx.DiGraph) and "html" (str) and "stats" (dict)
#
# Stored here (module level) so all routes can import and use it.
# graph_cache is imported by routes/graph.py, routes/bugs.py, etc.
# -----------------------------------------------------------------------
graph_cache: dict = {}


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a GitHub repository",
    description="Clone, parse, embed, and index a public GitHub repository for analysis.",
)
async def ingest_repo(request: IngestRequest):
    """
    Full ingestion pipeline for a GitHub repository.

    Steps:
      1. Clone repo
      2. Walk + filter source files
      3. Chunk files via AST
      4. Embed + store in Qdrant
      5. Build dependency graph
      6. Clean up disk

    Returns summary stats on success.
    Returns 400 on bad URL, 500 on pipeline failure.
    """

    # ── Step 1: Clone ──────────────────────────────────────────────────
    print(f"\n[ingest] Starting ingestion for: {request.github_url}")
    clone_result = clone_repo(request.github_url)

    if clone_result["status"] == "error":
        # 400 Bad Request — invalid URL or private repo
        raise HTTPException(
            status_code=400,
            detail=clone_result["message"],
        )

    repo_name  = clone_result["repo_name"]
    local_path = clone_result["local_path"]
    print(f"[ingest] Cloned '{repo_name}' to {local_path}")

    try:
        # ── Step 2: Walk ───────────────────────────────────────────────
        file_list = walk_repo(local_path)

        if not file_list:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No supported source files found in '{repo_name}'. "
                    f"Supported: .py, .js, .ts, .jsx, .tsx"
                ),
            )

        print(f"[ingest] Found {len(file_list)} source files")

        # ── Step 3: Chunk ──────────────────────────────────────────────
        all_chunks = chunk_files(file_list)

        if not all_chunks:
            raise HTTPException(
                status_code=500,
                detail="Chunking produced no output. This is unexpected — check logs.",
            )

        print(f"[ingest] Produced {len(all_chunks)} chunks total")

        # ── Step 4: Embed + Store ──────────────────────────────────────
        embed_result = embed_and_store(all_chunks, repo_name)

        if embed_result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Embedding failed: {embed_result['message']}",
            )

        print(f"[ingest] Stored {embed_result['chunks_stored']} chunks in Qdrant")

        # ── Step 5: Build Dependency Graph ─────────────────────────────
        # Build graph while local files still exist (renderer needs paths)
        graph = build_dependency_graph(file_list)
        stats = get_graph_stats(graph)

        # Store in cache for other routes to access
        graph_cache[repo_name] = {
            "graph": graph,
            "stats": stats,
        }

        print(f"[ingest] Graph built: {stats['nodes']} nodes, {stats['edges']} edges")

    finally:
        # ── Step 6: Clean up clone ─────────────────────────────────────
        # Always runs — even if an exception was raised above.
        # This prevents disk accumulation on Railway even after errors.
        try:
            shutil.rmtree(local_path, onerror=_force_remove_readonly)
            print(f"[ingest] Cleaned up {local_path}")
        except Exception:
            pass  # cleanup failure is non-critical

    return IngestResponse(
        status="success",
        repo_name=repo_name,
        files_indexed=len(file_list),
        chunks_stored=embed_result["chunks_stored"],
        graph_ready=True,
        message=f"'{repo_name}' indexed successfully. Ready for analysis.",
    )