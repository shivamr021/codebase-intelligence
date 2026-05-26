"""
main.py — FastAPI application entry point.

This file's only jobs:
  1. Create the FastAPI app instance
  2. Add CORS middleware (so the frontend on Vercel can talk to this backend)
  3. Validate config at startup (fail fast if API keys are missing)
  4. Register all API routers (we'll add these on Day 6)
  5. Provide a health check endpoint

What main.py does NOT do:
  - No business logic here
  - No direct database calls
  - No LLM calls
  Everything is in core/ and api/routes/

Why this matters in interviews:
  "Separation of concerns" — the entry point just wires things up.
  If something breaks, you know immediately which layer to look at.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import validate_config

# -----------------------------------------------------------------------
# FastAPI app instance
#
# title, description, version appear in the auto-generated Swagger docs
# at http://localhost:8000/docs — this is what you show interviewers.
# -----------------------------------------------------------------------
app = FastAPI(
    title="Codebase Intelligence System",
    description=(
        "Upload any public GitHub repo and get: architecture diagrams, "
        "dependency graphs, bug detection, and natural language Q&A about the codebase."
    ),
    version="1.0.0",
)


# -----------------------------------------------------------------------
# CORS Middleware — Critical for frontend communication
#
# CORS (Cross-Origin Resource Sharing) is a browser security rule.
# When your React frontend (on vercel.app) calls your backend (on railway.app),
# the browser blocks the request unless the backend explicitly says
# "I allow requests from that origin."
#
# allow_origins: which frontend URLs are allowed to call this backend.
#   ["*"] means everyone — fine for a portfolio project, not for production.
#   In production you'd list your specific Vercel URL.
#
# allow_methods: which HTTP methods are allowed (GET, POST, etc.)
#
# allow_headers: which request headers are allowed.
#   ["*"] allows everything including "Content-Type" which fetch() needs.
#
# Without this middleware, your frontend will get a CORS error and
# every API call will fail silently in the browser. This is one of the
# most common "why isn't it working" issues in web development.
# -----------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------
# Startup event — runs ONCE when the server first starts
#
# @app.on_event("startup") is a FastAPI lifecycle hook.
# Code here runs after the app is created but before it accepts requests.
#
# We use it to:
#   1. Validate that all required env vars are present
#   2. (Later) warm up the Qdrant connection
#
# If validate_config() raises ValueError, the server crashes immediately
# with a clear message. Better than crashing on the first real request
# with a confusing KeyError deep in the code.
# -----------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Runs once at server startup. Validates configuration."""
    validate_config()
    print("✓ Config validated — all environment variables present")
    print("✓ Codebase Intelligence System is ready")


# -----------------------------------------------------------------------
# Health check endpoint — GET /
#
# This is the endpoint your Railway cron job pings to keep the server warm.
# Also useful for quickly checking "is the server up?" without any logic.
#
# Returns a simple JSON response. Railway and any monitoring tool
# considers a 200 response as "healthy".
#
# In the auto-generated Swagger docs this appears as the first endpoint.
# -----------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health_check():
    """
    Health check. Returns 200 if the server is running.
    Use this URL for uptime monitoring or cron job pings.
    """
    return {
        "status": "running",
        "service": "Codebase Intelligence System",
        "version": "1.0.0",
        "docs": "/docs",
    }


# -----------------------------------------------------------------------
# Test cloner endpoint — POST /test-clone
#
# Temporary endpoint for Day 1 testing ONLY.
# Lets you test cloner.py directly without building the full pipeline.
# We will DELETE this before deploying to Railway.
#
# Try it at: http://localhost:8000/docs → POST /test-clone
# Body: { "github_url": "https://github.com/tiangolo/fastapi" }
# -----------------------------------------------------------------------
from pydantic import BaseModel  # noqa: E402 — import here to keep it obviously temporary

class TestCloneRequest(BaseModel):
    github_url: str

@app.post("/test-clone", tags=["Day 1 Testing — DELETE BEFORE DEPLOY"])
async def test_clone(request: TestCloneRequest):
    """
    Temporary endpoint to test cloner.py.
    DELETE THIS before deploying.
    """
    from core.ingestion.cloner import clone_repo
    result = clone_repo(request.github_url)
    return result


# -----------------------------------------------------------------------
# ROUTERS — registered here as we build each day
#
# Commented out for now. Uncomment each line as you complete that day.
# Pattern:
#   from api.routes.ingest import router as ingest_router
#   app.include_router(ingest_router, prefix="/api/v1", tags=["Ingestion"])
# -----------------------------------------------------------------------

# Day 6 — uncomment as you build each route:
# from api.routes.ingest import router as ingest_router
# from api.routes.query import router as query_router
# from api.routes.graph import router as graph_router
# from api.routes.bugs import router as bugs_router
# from api.routes.diagram import router as diagram_router

# app.include_router(ingest_router, prefix="/api/v1", tags=["Ingestion"])
# app.include_router(query_router, prefix="/api/v1", tags=["Query"])
# app.include_router(graph_router, prefix="/api/v1", tags=["Graph"])
# app.include_router(bugs_router, prefix="/api/v1", tags=["Bugs"])
# app.include_router(diagram_router, prefix="/api/v1", tags=["Diagram"])