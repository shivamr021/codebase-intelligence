"""
core/embeddings/vectors.py — vector_space: the one owner of model and vector facts (AD10).

Dense only for now: the vector name, the lazy model load from the baked cache,
the thread count, point vectors and query vectors. P4 adds the tokenizer
functions, P11 the sparse vectors.
"""

import threading
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.text.text_embedding import TextEmbedding as _Catalog  # the real class: model descriptions only

from config import EMBED_CACHE_DIR, EMBED_THREADS, EMBEDDING_MODEL

_model = None
_lock = threading.Lock()


def dense_name() -> str:
    """The name qdrant-client 1.9.1 gave the vector ("fast-<model>"), so old collections stay readable."""
    return f"fast-{EMBEDDING_MODEL.split('/')[-1].lower()}"


def _dense_model():
    """Load the configured model once, on first use, never at import; only from the baked cache."""
    global _model
    with _lock:
        if _model is None:
            _model = TextEmbedding(EMBEDDING_MODEL, cache_dir=EMBED_CACHE_DIR, threads=EMBED_THREADS,
                                   local_files_only=True)
    return _model


def passage_vectors(texts: list[str]) -> list[list[float]]:
    """Dense vectors for stored chunks."""
    return [vector.tolist() for vector in _dense_model().passage_embed(texts)]


def query_vector(text: str) -> list[float]:
    """Dense vector for one question."""
    return next(iter(_dense_model().query_embed(text))).tolist()


def require_model_cache() -> None:
    """Fail startup when the baked model files are missing; never download them (ruling 4)."""
    desc = next((m for m in _Catalog.list_supported_models() if m["model"] == EMBEDDING_MODEL), None)
    if desc:  # Hugging Face cache layout: models--<org>--<repo>/refs/main names the snapshot folder
        repo = Path(EMBED_CACHE_DIR, "models--" + desc["sources"]["hf"].replace("/", "--"))
        ref = repo / "refs" / "main"
        files = [desc["model_file"], *desc["additional_files"], "tokenizer.json", "config.json"]
        if ref.is_file() and all((repo / "snapshots" / ref.read_text().strip() / f).is_file() for f in files):
            return
    raise RuntimeError(
        f"Embedding model {EMBEDDING_MODEL} is missing from the model cache {EMBED_CACHE_DIR}. "
        f"Models are baked into the image at build time and never downloaded at run time: "
        f"rebuild the image, or set EMBED_CACHE_DIR to a folder that holds the model."
    )
