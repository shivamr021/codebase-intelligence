"""tests/conftest.py — shared harness (B1a): one in-memory Qdrant, dummy env, fake Groq server, fake model loader."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import dotenv
import fastembed
import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from qdrant_client import QdrantClient, qdrant_fastembed

DUMMY_ENV = {"GROQ_API_KEY": "test-dummy-groq-key", "QDRANT_URL": "http://qdrant.invalid:6333",
             "QDRANT_API_KEY": "test-dummy-qdrant-key"}
UNSET_ENV = ("GROQ_MODEL", "GROQ_MAX_TOKENS", "EMBEDDING_MODEL", "TMP_DIR", "GITHUB_TOKEN", "EMBED_THREADS", "FASTEMBED_CACHE_PATH")
QDRANT_ATTRS = {"core.embeddings.embedder": "_qdrant", "core.retrieval.retriever": "_qdrant",
                "core.storage.repo_metadata": "_client", "api.routes.check": "_qdrant"}
CHUNKS = [{"file_path": f"src/{n}.py", "name": n, "text": f"def {n}(): return {i}", "chunk_type": "function",
           "start_line": 1, "end_line": 1, "language": "python"} for i, n in enumerate(["alpha", "beta"])]

DIMS = {"BAAI/bge-small-en-v1.5": 384, "BAAI/bge-base-en-v1.5": 768, "BAAI/bge-small-en": 384}  # fastembed model list
HF_REPOS = {"BAAI/bge-small-en-v1.5": "Qdrant/bge-small-en-v1.5-onnx-Q", "BAAI/bge-base-en-v1.5": "Qdrant/bge-base-en-v1.5-onnx-Q"}
MODEL_FILES = ("model_optimized.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

def snapshot(cache, model):  # fastembed 0.9.1 Hugging Face cache layout: models--<org>--<repo>/snapshots/<rev>/
    return Path(cache, "models--" + HF_REPOS.get(model, "none/none").replace("/", "--"), "snapshots", "0" * 40)

class FakeTextEmbedding:  # fastembed.TextEmbedding, the model-download boundary: records loads, threads, downloads
    loads: list = []
    inits: list = []
    downloads: list = []
    def __init__(self, model_name, cache_dir=None, threads=None, local_files_only=False, **kwargs):
        if not (cache_dir and all((snapshot(cache_dir, model_name) / f).is_file() for f in MODEL_FILES)):
            if local_files_only:
                raise ValueError("fake fastembed: model files missing and local_files_only=True")
            FakeTextEmbedding.downloads.append(model_name)  # the real class fetches the model files here
        self.dim = DIMS[model_name]
        FakeTextEmbedding.loads.append(model_name)
        FakeTextEmbedding.inits.append((model_name, threads))
    embed = passage_embed = lambda self, documents, **kwargs: (np.full(self.dim, 0.1) for _ in documents)  # noqa: E731
    query_embed = lambda self, query, **kwargs: self.embed([query])  # noqa: E731

class FakeGroq:  # the Groq server behind OpenAI(...).chat.completions: records each request
    def __init__(self):
        self.requests, self.chat = [], SimpleNamespace(completions=SimpleNamespace(create=self._create))
    def _create(self, **kwargs):
        self.requests.append(kwargs)
        choice = Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="fake reply"))
        return ChatCompletion(id="chatcmpl-1", object="chat.completion", created=0, model=kwargs["model"], choices=[choice])

@pytest.fixture
def shell_env():  # env already in the shell before the harness runs; a test overrides it by parametrizing "shell_env"
    return {}

@pytest.fixture(scope="session")
def model_cache(tmp_path_factory):  # X3: the baked cache as the Dockerfile leaves it; EMBED_CACHE_DIR points here
    cache = tmp_path_factory.mktemp("model-cache")
    for model in HF_REPOS:
        snapshot(cache, model).mkdir(parents=True)
        for name in MODEL_FILES:
            (snapshot(cache, model) / name).write_text("")  # names only; the fake model never reads them
        (snapshot(cache, model).parents[1] / "refs").mkdir()
        (snapshot(cache, model).parents[1] / "refs" / "main").write_text("0" * 40)
    return cache

def bind_fake_model(monkeypatch):  # every lookup of the model class: 1.9.1's mixin, fastembed, our vector_space
    monkeypatch.setattr(qdrant_fastembed, "TextEmbedding", FakeTextEmbedding, raising=False)
    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    if importlib.util.find_spec("core.embeddings.vectors"):  # P7 creates it; reload resets its lazy model per test
        vectors = importlib.reload(importlib.import_module("core.embeddings.vectors"))
        monkeypatch.setattr(vectors, "TextEmbedding", FakeTextEmbedding, raising=False)

@pytest.fixture(autouse=True)
def harness(shell_env, monkeypatch, model_cache):
    for key, value in {**shell_env, **DUMMY_ENV, "EMBED_CACHE_DIR": str(model_cache)}.items():
        monkeypatch.setenv(key, value)
    for key in UNSET_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)  # a local .env must not refill UNSET_ENV
    monkeypatch.setattr(qdrant_fastembed.QdrantFastembedMixin, "embedding_models", {}, raising=False)
    for record in ("loads", "inits", "downloads"):
        monkeypatch.setattr(FakeTextEmbedding, record, [])
    h = SimpleNamespace(store=QdrantClient(":memory:"), groq=FakeGroq(), loads=FakeTextEmbedding.loads, chunks=CHUNKS,
                        inits=FakeTextEmbedding.inits, downloads=FakeTextEmbedding.downloads)
    reloaded = []

    def patch(name, store=None):  # point a module's Qdrant client at `store` (default: the shared one); Groq at fake
        module = importlib.import_module(name)
        if name in QDRANT_ATTRS:
            monkeypatch.setattr(module, QDRANT_ATTRS[name], store or h.store)
        if name == "core.llm.client":
            monkeypatch.setattr(module, "_client", h.groq)

    def reload(*names, env=None):  # set env, reload config then each module so copied values see it; re-patch
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        importlib.reload(importlib.import_module("config"))
        bind_fake_model(monkeypatch)
        for name in names:
            importlib.reload(importlib.import_module(name))
            reloaded.append(name)
            patch(name)

    h.patch, h.reload = patch, reload
    reload()
    list(map(patch, (*QDRANT_ATTRS, "core.llm.client")))
    yield h
    monkeypatch.undo()
    vectors = ["core.embeddings.vectors"] if importlib.util.find_spec("core.embeddings.vectors") else []
    for name in ["config", *vectors, *dict.fromkeys(reloaded)]:
        importlib.reload(importlib.import_module(name))
