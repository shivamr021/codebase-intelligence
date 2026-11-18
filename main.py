"""
main.py — FastAPI application entry point.

Registers all routers. Adds CORS. Validates config at startup.
"""

import traceback

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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

def _log_error(request: Request, status: int, detail) -> None:
    # Method, path, status, reason only: never headers, bodies, queries or env.
    print(f"[error] {request.method} {request.url.path} -> {status}: {detail}", flush=True)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    _log_error(request, exc.status_code, exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    _log_error(request, 422, exc.errors())
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception):
    _log_error(request, 500, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


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
from api.routes.check import router as check

app.include_router(ingest_router,  prefix="/api/v1", tags=["Ingestion"])
app.include_router(query_router,   prefix="/api/v1", tags=["Query"])
app.include_router(graph_router,   prefix="/api/v1", tags=["Graph"])
app.include_router(bugs_router,    prefix="/api/v1", tags=["Bugs"])
app.include_router(diagram_router, prefix="/api/v1", tags=["Diagram"])
app.include_router(check,   prefix="/api/v1", tags=["Check"])