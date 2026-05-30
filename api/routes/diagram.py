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
from core.llm.client import call_llm
from core.llm.prompts import EXPLAIN_ARCHITECTURE, format_chunks_for_prompt

router = APIRouter()


def _parse_architecture_response(llm_response: str) -> tuple[str, str]:
    """
    Split the LLM response into:
      - summary: the text before the mermaid block
      - mermaid: just the diagram syntax (without the ``` fences)

    Returns ("summary text", "mermaid syntax") tuple.
    If no mermaid block found, returns full response as summary + empty mermaid.
    """
    # Look for ```mermaid ... ``` block
    mermaid_start = llm_response.find("```mermaid")
    mermaid_end   = llm_response.find("```", mermaid_start + 10)  # +10 to skip opening fence

    if mermaid_start == -1:
        # No mermaid block found — return full text as summary
        return llm_response.strip(), ""

    # Extract summary (everything before the mermaid block)
    summary = llm_response[:mermaid_start].strip()

    # Extract mermaid syntax (between the fences, strip the ```mermaid line)
    mermaid_raw = llm_response[mermaid_start:mermaid_end + 3]
    # Remove opening ```mermaid and closing ``` to get just the diagram syntax
    mermaid_lines = mermaid_raw.split("\n")
    # First line is ```mermaid, last line is ``` — remove both
    mermaid_syntax = "\n".join(mermaid_lines[1:-1]).strip()

    return summary, mermaid_syntax


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

    # Use broad sample for architecture — want overall picture, not specific detail
    chunks = retrieve_all_chunks(repo_name, limit=40)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed data for '{repo_name}'. Run /ingest first.",
        )

    code_context = format_chunks_for_prompt(chunks)

    files = sorted({
            chunk["file_path"]
            for chunk in chunks
    })

    repo_structure = "\n".join(files)

    prompt = EXPLAIN_ARCHITECTURE.format(
        repo_structure=repo_structure,
        code_context=code_context,
    )

    # Slightly higher temperature for architecture — allow more natural language
    llm_response = call_llm(prompt, temperature=0.4)

    summary, mermaid = _parse_architecture_response(llm_response)

    # Fallback mermaid if parsing produced nothing
    if not mermaid:
        mermaid = "graph TD\n    A[Could not generate diagram] --> B[Try re-ingesting the repo]"

    return DiagramResponse(
        status="success",
        repo_name=repo_name,
        summary=summary,
        mermaid=mermaid,
    )