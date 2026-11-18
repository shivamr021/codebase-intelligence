"""
api/routes/ingest.py — POST /api/v1/ingest

This is the entry point for the entire system.
Supports two indexing modes:

  1. FULL INDEX (first time):
     resolve → check SHA → clone → walk → chunk → embed → graph → store

  2. INCREMENTAL UPDATE (repo already indexed, new commit detected):
     resolve → check SHA → clone → walk → diff → chunk changed files →
     delete stale chunks → upsert new chunks → rebuild graph → update metadata

  3. UP-TO-DATE (repo already indexed, same commit):
     resolve → check SHA → return immediately (NO clone, NO download)

Key optimization (addressing early-abort):
  resolve_repo() only does URL validation + a single GitHub API call (~100ms).
  We compare the returned commit_sha against stored metadata BEFORE calling
  clone_repo() (which downloads the full zip). This saves bandwidth on the
  common case where the repo hasn't changed.

Graph rebuild on incremental update:
  build_dependency_graph() takes the FULL file_list from walker.py and reads
  each file from disk to parse import statements. Since the full clone is
  still on disk during the incremental path, the graph builder has access
  to ALL files (changed + unchanged). The graph is always built from the
  complete file tree — only the Qdrant embedding step is incremental.
"""

import shutil
import stat
import threading
from fastapi import APIRouter, HTTPException
from core.llm.context_selector import (
    select_architecture_chunks,
)
from core.graph.builder import (
    build_dependency_graph,
    get_graph_stats,
    get_architecture_context,
)

def _force_remove_readonly(func, path, _):
    """Windows read-only file handler for shutil.rmtree — see cloner.py for explanation."""
    import os
    os.chmod(path, stat.S_IWRITE)
    func(path)


from api.models import IngestRequest, IngestResponse, ErrorResponse
from core.ingestion.cloner import resolve_repo, clone_repo
from core.ingestion.walker import walk_repo
from core.ingestion.chunker import INDEX_VERSION, chunk_files, read_index_version
from core.ingestion.diff import compute_file_diff
from core.embeddings.embedder import (
    embed_and_store,
    upsert_chunks,
    delete_file_chunks,
)
from core.graph.builder import (
    build_dependency_graph,
    get_graph_stats,
)

from core.graph.serializer import (
    deserialize_graph,
    serialize_graph,
)
from core.retrieval.retriever import retrieve_all_chunks

from core.llm.architecture import (
    ARCHITECTURE_FAILED,
    generate_architecture,
)

from core.storage.repo_metadata import (
    save_repo_metadata,
    get_repo_metadata,
    is_index_compatible,
)

router = APIRouter()

# -----------------------------------------------------------------------
# In-memory graph cache.
# Key:   collection_name (str) — may include branch/commit suffix
# Value: dict with "graph" (nx.DiGraph) and "html" (str) and "stats" (dict)
# -----------------------------------------------------------------------
graph_cache: dict = {}


def _build_collection_name(
    repo_name: str,
    branch: str | None = None,
    commit_hash: str | None = None,
) -> str:
    """
    Build a versioned Qdrant collection name.

    Examples:
      - repo_name="fastapi", branch=None, commit=None → "fastapi"
      - repo_name="fastapi", branch="develop"          → "fastapi@develop"
      - repo_name="fastapi", commit_hash="abc123..."   → "fastapi@abc123de"

    The "@" separator cannot appear in a GitHub owner or repo name, so a
    versioned name never collides with a plain one; commit wins over branch.
    """
    if commit_hash:
        return f"{repo_name}@{commit_hash[:8]}"
    elif branch:
        # Sanitize branch name for Qdrant (no slashes, spaces)
        safe_branch = branch.replace("/", "-").replace(" ", "-")
        return f"{repo_name}@{safe_branch}"
    return repo_name


def _build_file_hashes(file_list: list[dict]) -> dict[str, str]:
    """Build a {rel_path: content_hash} map from walker output."""
    return {f["rel_path"]: f["content_hash"] for f in file_list}


def _run_graph_and_architecture(
    file_list, all_chunks, repo_name, graph_cache,
    commit_sha, file_hashes, index_version,
):
    """
    Shared logic for graph building + architecture generation + metadata save.
    Used by both full and incremental paths.

    index_version is the version of the chunks stored in Qdrant: the full
    path passes INDEX_VERSION; the incremental path keeps the stored one,
    because it rebuilt only the changed files' chunks.

    IMPORTANT: build_dependency_graph() takes the FULL file_list and reads
    each file from disk to parse imports. It does NOT use chunks.
    This means it always builds the complete dependency graph, even during
    incremental updates — because the full clone is still on disk at this point.

    The chunks are only used for the architecture LLM summary.
    """
    # Build dependency graph from ALL files on disk
    graph = build_dependency_graph(file_list)
    stats = get_graph_stats(graph)

    architecture_context = get_architecture_context(graph)

    # Generate architecture summary
    print(f"[ingest] Generating architecture for '{repo_name}'")

    architecture_chunks = select_architecture_chunks(
        all_chunks,
        max_chunks=12,
    )

    print(f"[ingest] Architecture context: {len(architecture_chunks)} chunks")

    for c in architecture_chunks:
        print(f"[ARCH] {c['file_path']}")

    summary, mermaid = generate_architecture(
        architecture_chunks,
        architecture_context,
    )

    architecture_ok = summary != ARCHITECTURE_FAILED
    print(f"[ingest] Architecture {'generated' if architecture_ok else 'failed'}")

    # AD13: the graph, hashes and SHA are saved even when the architecture
    # failed (architecture_ok: false), so /graph survives a restart and the
    # next ingest at the same SHA regenerates only the architecture.
    save_repo_metadata(
        repo_name=repo_name,
        summary=summary if architecture_ok else "",
        mermaid=mermaid if architecture_ok else "",
        graph_stats=stats,
        graph_data=serialize_graph(graph),
        architecture_context=architecture_context,
        commit_sha=commit_sha,
        file_hashes=file_hashes,
        extra={"index_version": index_version, "architecture_ok": architecture_ok},
    )
    print(f"[ingest] Metadata persisted (architecture_ok={architecture_ok})")

    # Store in cache for other routes to access
    graph_cache[repo_name] = {
        "graph": graph,
        "stats": stats,
    }

    print(f"[ingest] Graph built: {stats['nodes']} nodes, {stats['edges']} edges")

    return architecture_ok


def _with_architecture_note(message: str, architecture_ok: bool) -> str:
    """Tell the user when the architecture step failed, so they can re-ingest."""
    if architecture_ok:
        return message
    return f"{message} Architecture generation failed; ingest again to retry it."


def _architecture_ok(metadata: dict) -> bool:
    """A missing flag (metadata saved before AD13) counts as ok when an architecture is stored."""
    flag = metadata.get("architecture_ok")
    if flag is None:
        return bool(metadata.get("mermaid"))
    return bool(flag)


def _retry_architecture(collection_name, metadata, commit_sha, file_hashes) -> bool:
    """
    AD13: regenerate only the architecture for an index whose chunks and
    graph are already stored. No download and no embedding: the chunks
    come from Qdrant and the graph from the saved graph_data.
    """
    print(f"[ingest] Retrying only the architecture for '{collection_name}'")
    context = metadata.get("architecture_context") or get_architecture_context(
        deserialize_graph(metadata.get("graph_data") or {})
    )
    chunks = select_architecture_chunks(retrieve_all_chunks(collection_name, limit=100), max_chunks=12)
    summary, mermaid = generate_architecture(chunks, context)
    architecture_ok = summary != ARCHITECTURE_FAILED

    save_repo_metadata(
        repo_name=collection_name,
        summary=summary if architecture_ok else "",
        mermaid=mermaid if architecture_ok else "",
        graph_stats=metadata.get("graph_stats", {}),
        graph_data=metadata.get("graph_data", {}),
        architecture_context=context,
        commit_sha=commit_sha,
        file_hashes=file_hashes,
        extra={"index_version": read_index_version(metadata), "architecture_ok": architecture_ok},
    )
    print(f"[ingest] Architecture retry {'succeeded' if architecture_ok else 'failed again'}")
    return architecture_ok


# -----------------------------------------------------------------------
# One ingest per collection at a time (AD6). Different collections run
# side by side; a second ingest of a busy collection gets 409 at once.
# Routes are plain `def`, so FastAPI runs each one in its thread pool.
# -----------------------------------------------------------------------
_busy_collections: set[str] = set()
_busy_lock = threading.Lock()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a GitHub repository",
    description=(
        "Clone, parse, embed, and index a public GitHub repository for analysis. "
        "If the repo was previously indexed, performs an incremental update — "
        "only re-processing changed files. Optionally specify a branch or commit_hash "
        "to index a specific version."
    ),
)
def ingest_repo(request: IngestRequest):
    """
    Full ingestion pipeline for a GitHub repository.

    Three modes:
      1. First time   → full index (clone → walk → chunk → embed → graph)
      2. Same commit  → return "already up to date" (NO download)
      3. New commit   → incremental (diff → chunk changed → upsert/delete)

    Optional: pass branch or commit_hash to index a specific version.
    """

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: RESOLVE (cheap — no download)
    # Validate URL, parse owner/repo, resolve commit SHA via GitHub API.
    # If commit_hash is provided, validates it exists (hard 400 on invalid).
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[ingest] Starting ingestion for: {request.github_url}")
    if request.branch:
        print(f"[ingest] Targeting branch: {request.branch}")
    if request.commit_hash:
        print(f"[ingest] Targeting commit: {request.commit_hash[:12]}...")

    try:
        resolve_result = resolve_repo(
            request.github_url,
            branch=request.branch,
            commit_hash=request.commit_hash,
        )
    except ValueError as e:
        # validate_commit_hash or fetch_branch_head raised — hard 400
        raise HTTPException(status_code=400, detail=str(e))

    if resolve_result["status"] == "error":
        raise HTTPException(
            status_code=400,
            detail=resolve_result["message"],
        )

    repo_name  = resolve_result["repo_name"]
    commit_sha = resolve_result["commit_sha"]
    owner      = resolve_result["owner"]
    repo       = resolve_result["repo"]
    branch     = resolve_result.get("branch")

    # Build versioned collection name
    collection_name = _build_collection_name(repo_name, branch, request.commit_hash)

    print(f"[ingest] Resolved '{owner}/{repo}'")
    print(f"[ingest] Commit: {commit_sha[:12] + '...' if commit_sha else 'unknown'}")
    print(f"[ingest] Collection: '{collection_name}'")

    with _busy_lock:
        if collection_name in _busy_collections:
            raise HTTPException(
                status_code=409,
                detail=f"'{collection_name}' is already being indexed. Try again when it finishes.",
            )
        _busy_collections.add(collection_name)
    try:
        return _ingest_collection(request, repo_name, commit_sha, owner, repo, collection_name)
    finally:
        with _busy_lock:
            _busy_collections.discard(collection_name)


def _ingest_collection(request, repo_name, commit_sha, owner, repo, collection_name):
    """Phases 1b-5 of ingest_repo, run while this collection's lock is held."""
    # Check if this version has been indexed before
    existing_metadata = get_repo_metadata(collection_name)
    stored_sha = existing_metadata.get("commit_sha") if existing_metadata else None
    stored_hashes = existing_metadata.get("file_hashes", {}) if existing_metadata else {}

    # ══════════════════════════════════════════════════════════════════
    # FULL RE-INDEX: asked for with force, or the stored index was built
    # with another embedding model (old vector name) or an older chunker
    # (index_version). Neither early path may reuse such an index (AD4).
    # Nothing is deleted here: embed_and_store() replaces the collection
    # only after the download succeeded.
    # ══════════════════════════════════════════════════════════════════
    full_reindex = bool(existing_metadata) and (
        request.force or not is_index_compatible(collection_name, existing_metadata)
    )
    if full_reindex:
        reason = "force requested" if request.force else "stored index is not compatible"
        print(f"[ingest] Full re-index of '{collection_name}': {reason}")

    # ══════════════════════════════════════════════════════════════════
    # FAST PATH: Already up to date — NO download needed
    # We compare SHAs BEFORE calling clone_repo() to save bandwidth.
    # ══════════════════════════════════════════════════════════════════
    if (
        not full_reindex
        and commit_sha
        and stored_sha
        and commit_sha == stored_sha
    ):
        print(f"[ingest] '{collection_name}' is already up to date (commit {commit_sha[:12]})")
        print(f"[ingest] Skipping download — no bandwidth used")

        message = f"'{collection_name}' is already up to date (commit {commit_sha[:12]}...)."
        if not _architecture_ok(existing_metadata):
            architecture_ok = _retry_architecture(
                collection_name, existing_metadata, commit_sha, stored_hashes,
            )
            message = _with_architecture_note(message, architecture_ok)

        return IngestResponse(
            status="success",
            repo_name=collection_name,
            files_indexed=0,
            chunks_stored=0,
            graph_ready=collection_name in graph_cache,
            message=message,
            incremental=False,
            commit_sha=commit_sha,
        )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: CLONE (expensive — zip download)
    # Only reached if: first-time index OR commit SHA differs.
    # ══════════════════════════════════════════════════════════════════
    clone_result = clone_repo(
        request.github_url, owner, repo,
        branch=request.branch,
        commit_hash=request.commit_hash,
        resolved_sha=commit_sha,
    )

    if clone_result["status"] == "error":
        raise HTTPException(
            status_code=400,
            detail=clone_result["message"],
        )

    local_path = clone_result["local_path"]
    work_dir = clone_result["work_dir"]  # this run's own folder (AD11)
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

        # Build current file hash map
        current_hashes = _build_file_hashes(file_list)

        # ── Decide: full index or incremental ──────────────────────────
        if existing_metadata and stored_hashes and not full_reindex:
            # ══════════════════════════════════════════════════════════
            # INCREMENTAL PATH
            # ══════════════════════════════════════════════════════════
            print(f"[ingest] Previous index found (commit {stored_sha[:12] if stored_sha else 'unknown'}...)")
            print(f"[ingest] Running incremental diff...")

            diff = compute_file_diff(file_list, stored_hashes)

            n_added = len(diff["added"])
            n_modified = len(diff["modified"])
            n_deleted = len(diff["deleted"])
            n_unchanged = len(diff["unchanged"])

            print(
                f"[ingest] Diff result: "
                f"{n_added} added, {n_modified} modified, "
                f"{n_deleted} deleted, {n_unchanged} unchanged"
            )

            # If nothing changed (commit SHA differed but file content is identical)
            if n_added == 0 and n_modified == 0 and n_deleted == 0:
                print(f"[ingest] No file content changes detected — skipping re-index")
                # Still update the commit SHA so next check is faster
                message = f"'{collection_name}' content unchanged — metadata updated."
                if not _architecture_ok(existing_metadata):
                    architecture_ok = _retry_architecture(
                        collection_name, existing_metadata, commit_sha, current_hashes,
                    )
                    message = _with_architecture_note(message, architecture_ok)
                elif commit_sha:
                    save_repo_metadata(
                        repo_name=collection_name,
                        summary=existing_metadata.get("summary", ""),
                        mermaid=existing_metadata.get("mermaid", ""),
                        graph_stats=existing_metadata.get("graph_stats", {}),
                        graph_data=existing_metadata.get("graph_data", {}),
                        architecture_context=existing_metadata.get("architecture_context"),
                        commit_sha=commit_sha,
                        file_hashes=current_hashes,
                        extra={"index_version": read_index_version(existing_metadata), "architecture_ok": True},
                    )
                return IngestResponse(
                    status="success",
                    repo_name=collection_name,
                    files_indexed=len(file_list),
                    chunks_stored=0,
                    graph_ready=collection_name in graph_cache,
                    message=message,
                    incremental=True,
                    files_added=0,
                    files_modified=0,
                    files_deleted=0,
                    commit_sha=commit_sha,
                )

            # ── Delete stale chunks ────────────────────────────────────
            # For modified files: delete old chunks first, then upsert new ones.
            # This is necessary because a modified file might produce different
            # chunks (function renamed, split, etc.)
            files_to_delete = diff["deleted"] + [f["rel_path"] for f in diff["modified"]]
            if files_to_delete:
                delete_file_chunks(collection_name, files_to_delete)
                print(f"[ingest] Deleted stale chunks for {len(files_to_delete)} files")

            # ── Chunk and upsert ONLY changed files ────────────────────
            files_to_chunk = diff["added"] + diff["modified"]
            new_chunks = chunk_files(files_to_chunk)

            chunks_stored = 0
            if new_chunks:
                upsert_result = upsert_chunks(
                    new_chunks, collection_name,
                    commit_hash=commit_sha,
                    collection_name=collection_name,
                )
                if upsert_result["status"] == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=f"Upsert failed: {upsert_result['message']}",
                    )
                chunks_stored = upsert_result["chunks_stored"]

            print(f"[ingest] Incremental update: {chunks_stored} chunks upserted")

            # ── Rebuild graph + architecture ───────────────────────────
            # Graph: build_dependency_graph() takes the FULL file_list and
            # reads ALL files from disk to parse import statements. Since
            # the full clone is still on disk, this always produces the
            # complete, correct global dependency graph.
            #
            # Architecture LLM: needs representative chunks. We chunk ALL
            # files for this (including unchanged) so the LLM sees the
            # full picture. These unchanged-file chunks are NOT sent to
            # Qdrant (they're already stored there from the previous run).
            all_chunks_for_architecture = new_chunks + chunk_files(diff["unchanged"])

            architecture_ok = _run_graph_and_architecture(
                file_list, all_chunks_for_architecture, collection_name, graph_cache,
                commit_sha, current_hashes, read_index_version(existing_metadata),
            )

            return IngestResponse(
                status="success",
                repo_name=collection_name,
                files_indexed=len(file_list),
                chunks_stored=chunks_stored,
                graph_ready=True,
                message=_with_architecture_note(
                    f"'{collection_name}' incrementally updated: "
                    f"{n_added} added, {n_modified} modified, {n_deleted} deleted.",
                    architecture_ok,
                ),
                incremental=True,
                files_added=n_added,
                files_modified=n_modified,
                files_deleted=n_deleted,
                commit_sha=commit_sha,
            )

        else:
            # ══════════════════════════════════════════════════════════
            # FULL INDEX PATH (first time or no stored hashes)
            # ══════════════════════════════════════════════════════════
            print(f"[ingest] No previous index — running full ingestion")

            # ── Step 3: Chunk ──────────────────────────────────────────
            all_chunks = chunk_files(file_list)

            if not all_chunks:
                raise HTTPException(
                    status_code=500,
                    detail="Chunking produced no output. This is unexpected — check logs.",
                )

            print(f"[ingest] Produced {len(all_chunks)} chunks total")

            # ── Step 4: Embed + Store ──────────────────────────────────
            embed_result = embed_and_store(
                all_chunks, collection_name,
                commit_hash=commit_sha,
                collection_name=collection_name,
            )

            if embed_result["status"] == "error":
                raise HTTPException(
                    status_code=500,
                    detail=f"Embedding failed: {embed_result['message']}",
                )

            print(f"[ingest] Stored {embed_result['chunks_stored']} chunks in Qdrant")

            # ── Step 5: Graph + Architecture + Metadata ────────────────
            architecture_ok = _run_graph_and_architecture(
                file_list, all_chunks, collection_name, graph_cache,
                commit_sha, current_hashes, INDEX_VERSION,
            )

            return IngestResponse(
                status="success",
                repo_name=collection_name,
                files_indexed=len(file_list),
                chunks_stored=embed_result["chunks_stored"],
                graph_ready=True,
                message=_with_architecture_note(
                    f"'{collection_name}' indexed successfully. Ready for analysis.",
                    architecture_ok,
                ),
                incremental=False,
                commit_sha=commit_sha,
            )

    finally:
        # ── Cleanup: remove cloned repo from disk ──────────────────────
        # Always runs — even if an exception was raised above.
        # This prevents disk accumulation on Railway even after errors.
        # Deletes only this run's folder, never another run's (AD11).
        try:
            shutil.rmtree(work_dir, onerror=_force_remove_readonly)
            print(f"[ingest] Cleaned up {work_dir}")
        except Exception:
            pass  # cleanup failure is non-critical