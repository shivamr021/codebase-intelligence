"""
api/routes/query.py — POST /api/v1/ask

Natural language Q&A about an indexed codebase.

Flow:
  1. Receive question + repo_name
  2. Retrieve top-5 relevant chunks from Qdrant (semantic search)
  3. Format chunks into prompt context
  4. Call Groq LLM with the ANSWER_QUESTION prompt
  5. Return answer with citation metadata
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.models import AskRequest, AskResponse, AskSource
from api.routes import ingest as ingest_routes  # module access: a reload rebinds its busy sets
from core.retrieval.retriever import retrieve_chunks
from core.llm.client import LLM_ERROR_PREFIX, call_llm_reply
from core.llm.prompts import ANSWER_QUESTION, REFUSAL_SENTENCE, format_chunks_for_prompt

router = APIRouter()


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the codebase",
    description="Retrieve relevant code chunks and generate an LLM answer with file citations.",
)
def ask_question(request: AskRequest):
    """
    Semantic search + LLM answer for any question about the codebase.

    The answer includes exact file paths and line numbers from the
    retrieved chunks — so every claim is verifiable.

    If commit_hash is provided, only chunks from that specific version
    are returned — enabling version-specific Q&A.
    """

    # AD11, ruling 1: a repo under a full re-index has a half-built index.
    # An incremental update keeps answering from the old index.
    if ingest_routes.is_rebuilding(request.repo_name):
        raise HTTPException(
            status_code=503,
            detail=f"'{request.repo_name}' is being re-indexed, try again in a minute.",
        )

    # Retrieve top-5 most relevant chunks for this question
    chunks = retrieve_chunks(
        query=request.question,
        repo_name=request.repo_name,
        top_k=5,
        commit_hash=request.commit_hash,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No indexed data found for repo '{request.repo_name}'"
                + (f" at commit {request.commit_hash[:12]}..." if request.commit_hash else "")
                + ". Run POST /api/v1/ingest first."
            ),
        )

    # Format chunks into the context string for the prompt
    code_context = format_chunks_for_prompt(chunks)

    # Build the full prompt with context injected
    prompt = ANSWER_QUESTION.format(
        code_context=code_context,
        question=request.question,
    )

    # Call Groq
    answer, cut_off = call_llm_reply(prompt, temperature=0.3)

    # An LLM failure is never a 200 answer. The error_code field lets
    # on-call and the frontend tell it apart from a proxy 502.
    if answer.startswith(LLM_ERROR_PREFIX):
        print(f"[error] POST /api/v1/ask -> 502: {answer}", flush=True)
        return JSONResponse(
            status_code=502,
            content={"detail": answer, "error_code": "LLM_UPSTREAM_ERROR"},
        )

    return AskResponse(
        status="success",
        repo_name=request.repo_name,
        question=request.question,
        answer=answer,
        chunks_used=len(chunks),
        sources=[
            AskSource(
                file=c["file_path"],
                name=c["name"],
                lines=f"{c['start_line']}-{c['end_line']}",
                score=c["score"],
            )
            for c in chunks
        ],
        refused=answer == REFUSAL_SENTENCE,
        cut_off=cut_off,
    )