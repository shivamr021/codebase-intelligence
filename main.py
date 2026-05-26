"""
main.py — FastAPI application entry point.

Registers all routers. Adds CORS. Validates config at startup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import validate_config

app = FastAPI(
    title="Codebase Intelligence System",
    description=(
        "Upload any public GitHub repo and get: architecture diagrams, "
        "dependency graphs, bug detection, and natural language Q&A about the codebase."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    validate_config()
    print("✓ Config validated — all environment variables present")
    print("✓ Codebase Intelligence System is ready")

@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "running",
        "service": "Codebase Intelligence System",
        "version": "1.0.0",
        "docs": "/docs",
    }

# ── Routers ────────────────────────────────────────────────────────────
from api.routes.ingest  import router as ingest_router
from api.routes.query   import router as query_router
from api.routes.graph   import router as graph_router
from api.routes.bugs    import router as bugs_router
from api.routes.diagram import router as diagram_router

app.include_router(ingest_router,  prefix="/api/v1", tags=["Ingestion"])
app.include_router(query_router,   prefix="/api/v1", tags=["Query"])
app.include_router(graph_router,   prefix="/api/v1", tags=["Graph"])
app.include_router(bugs_router,    prefix="/api/v1", tags=["Bugs"])
app.include_router(diagram_router, prefix="/api/v1", tags=["Diagram"])