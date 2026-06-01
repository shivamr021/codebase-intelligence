"""
api/routes/diagram.py — GET /api/v1/diagram?repo_name=fastapi

Generates an architecture summary and Mermaid diagram for the codebase.

Flow:
  1. Retrieve broad code sample (scroll, not query-specific)
  2. Run EXPLAIN_ARCHITECTURE prompt
  3. Parse response into: summary text + mermaid code block
  4. Return both to frontend

Mermaid parsing:
  The LLM returns a response like:
    "This codebase is a FastAPI application...

    ```mermaid
    graph TD
        A[main.py] --> B[auth.py]
    ```"

  We split on the ```mermaid fence to extract just the diagram syntax.
  The frontend passes this string to mermaid.js which renders it as SVG.
"""

from fastapi import APIRouter, HTTPException

from api.models import DiagramResponse
from core.retrieval.retriever import retrieve_all_chunks
from core.llm.architecture import (
    generate_architecture
)

from core.storage.repo_metadata import get_repo_metadata

router = APIRouter()


@router.get(
    "/diagram",
    response_model=DiagramResponse,
    summary="Generate architecture diagram",
    description="Generate a Mermaid architecture diagram and summary for the indexed repo.",
)
async def get_diagram(repo_name: str):
    """
    Returns architecture summary + Mermaid syntax.
    Frontend renders Mermaid string using mermaid.js — no image generation needed.
    """

    cached = get_repo_metadata(repo_name)

    if cached:
        print(
            f"[diagram.py] Using cached architecture "
            f"for '{repo_name}'"
        )

        return DiagramResponse(
            status="success",
            repo_name=repo_name,
            summary=cached["summary"],
            mermaid=cached["mermaid"],
        )

    # Use broad sample for architecture — want overall picture, not specific detail
    chunks = retrieve_all_chunks(repo_name, limit=10)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed data for '{repo_name}'. Run /ingest first.",
        )

    summary, mermaid = generate_architecture(
        chunks
    )

    return DiagramResponse(
        status="success",
        repo_name=repo_name,
        summary=summary,
        mermaid=mermaid,
    )