"""
api/routes/bugs.py — GET /api/v1/bugs?repo_name=fastapi

Retrieves a broad sample of code chunks and asks Groq to find bugs,
code smells, and issues. Returns structured JSON list.

Flow:
  1. Scroll top-30 chunks from Qdrant (broad coverage, not query-specific)
  2. Run FIND_BUGS prompt — asks for JSON array output
  3. Parse the JSON from the LLM response
  4. Return typed BugItem list

Why scroll() not query()?
  Bug detection is a broad-coverage task. We want to see many parts of
  the codebase, not just the parts semantically similar to one query.
  retrieve_all_chunks() uses Qdrant's scroll() for this.

JSON parsing risk:
  LLMs sometimes add text before/after the JSON array despite instructions.
  We handle this by finding the [ ... ] boundaries in the response string.
  If parsing still fails, we return a graceful error — never crash.
"""

import json
from fastapi import APIRouter, HTTPException

from api.models import BugsResponse, BugItem
from core.retrieval.retriever import retrieve_all_chunks
from core.llm.client import call_llm
from core.llm.prompts import FIND_BUGS, format_chunks_for_prompt

router = APIRouter()


def _parse_bug_json(llm_response: str) -> list[dict]:
    """
    Extract and parse the JSON array from the LLM response.

    The LLM is instructed to return ONLY a JSON array, but sometimes
    adds preamble like "Here are the bugs:" before the array.
    We find the first '[' and last ']' to extract just the JSON.

    A valid empty array [] means "no bugs found" and is returned as [].
    Raises ValueError with the reason when the reply is unusable, so the
    route answers 502 instead of a false "no issues detected".
    """
    if llm_response.startswith("[LLM Error]"):
        raise ValueError(f"Bug scan failed: {llm_response}")

    # Find the JSON array boundaries
    start = llm_response.find("[")
    end   = llm_response.rfind("]") + 1   # rfind = last occurrence

    if start == -1 or end == 0:
        print("[bugs.py] LLM did not return a JSON array.")
        raise ValueError("Bug scan failed: the model did not return a JSON array.")

    json_str = llm_response[start:end]
    try:
        bugs = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[bugs.py] JSON parse error: {e}")
        print(f"[bugs.py] Raw response: {llm_response[:500]}")
        raise ValueError(f"Bug scan failed: the model's JSON array could not be parsed ({e}).") from e

    # Validate it's actually a list
    if not isinstance(bugs, list):
        raise ValueError("Bug scan failed: the model did not return a JSON array.")

    return bugs


@router.get(
    "/bugs",
    response_model=BugsResponse,
    summary="Detect bugs and code issues",
    description="Analyse the codebase for bugs, code smells, and issues using LLM review.",
)
def detect_bugs(repo_name: str):
    """
    Broad-coverage bug detection across the indexed codebase.
    Returns up to 10 issues sorted by severity.
    """

    # Retrieve broad sample — 30 chunks from across the codebase
    chunks = retrieve_all_chunks(repo_name, limit=15)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed data for '{repo_name}'. Run /ingest first.",
        )

    # Format and build prompt
    code_context = format_chunks_for_prompt(chunks)
    prompt = FIND_BUGS.format(code_context=code_context)

    # Use lower temperature for bug detection — want consistent, deterministic output
    llm_response = call_llm(prompt, temperature=0.1)

    # Parse structured JSON from response; an unusable reply is a 502
    try:
        raw_bugs = _parse_bug_json(llm_response)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Convert to BugItem models — skip any malformed entries
    bug_items = []
    for bug in raw_bugs:
        try:
            bug_items.append(BugItem(
                file=str(bug.get("file", "unknown")),
                line=int(bug.get("line", 0)),
                severity=str(bug.get("severity", "low")),
                issue=str(bug.get("issue", "")),
                suggestion=str(bug.get("suggestion", "")),
            ))
        except Exception:
            continue  # Skip malformed bug entries silently

    return BugsResponse(
        status="success",
        repo_name=repo_name,
        bugs=bug_items,
        total=len(bug_items),
    )