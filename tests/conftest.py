"""tests/conftest.py — shared harness (B1a): one in-memory Qdrant, dummy env, fake Groq server, fake model loader."""
import importlib
from types import SimpleNamespace
import dotenv
import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from qdrant_client import QdrantClient, qdrant_fastembed

DUMMY_ENV = {"GROQ_API_KEY": "test-dummy-groq-key", "QDRANT_URL": "http://qdrant.invalid:6333",
             "QDRANT_API_KEY": "test-dummy-qdrant-key"}
UNSET_ENV = ("GROQ_MODEL", "GROQ_MAX_TOKENS", "EMBEDDING_MODEL", "TMP_DIR", "GITHUB_TOKEN")
QDRANT_ATTRS = {"core.embeddings.embedder": "_qdrant", "core.retrieval.retriever": "_qdrant",
                "core.storage.repo_metadata": "_client", "api.routes.check": "_qdrant"}
CHUNKS = [{"file_path": f"src/{n}.py", "name": n, "text": f"def {n}(): return {i}", "chunk_type": "function",
           "start_line": 1, "end_line": 1, "language": "python"} for i, n in enumerate(["alpha", "beta"])]

class FakeTextEmbedding:  # fastembed.TextEmbedding, the model-download boundary: records loads, fixed vectors
    loads: list = []
    def __init__(self, model_name, **kwargs):
        self.dim = qdrant_fastembed.SUPPORTED_EMBEDDING_MODELS[model_name][0]
        FakeTextEmbedding.loads.append(model_name)
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

@pytest.fixture(autouse=True)
def harness(shell_env, monkeypatch):
    for key, value in {**shell_env, **DUMMY_ENV}.items():
        monkeypatch.setenv(key, value)
    for key in UNSET_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)  # a local .env must not refill UNSET_ENV
    monkeypatch.setattr(qdrant_fastembed, "TextEmbedding", FakeTextEmbedding)
    monkeypatch.setattr(qdrant_fastembed.QdrantFastembedMixin, "embedding_models", {})
    monkeypatch.setattr(FakeTextEmbedding, "loads", [])
    h = SimpleNamespace(store=QdrantClient(":memory:"), groq=FakeGroq(), loads=FakeTextEmbedding.loads, chunks=CHUNKS)
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
        for name in names:
            importlib.reload(importlib.import_module(name))
            reloaded.append(name)
            patch(name)

    h.patch, h.reload = patch, reload
    reload()
    list(map(patch, (*QDRANT_ATTRS, "core.llm.client")))
    yield h
    monkeypatch.undo()
    for name in ["config", *dict.fromkeys(reloaded)]:
        importlib.reload(importlib.import_module(name))
