from collections.abc import Iterable, Iterator
from pathlib import Path


HIGH_PRIORITY_FILES = {
    "main.py",
    "app.py",
    "server.py",
    "index.ts",
    "index.js",
    "main.tsx",
    "app.tsx",
    "config.py",
}


NEGATIVE_FILES = {
    "eslint.config.js",
    "postcss.config.js",
    "tailwind.config.js",
}


def score_file(path: str) -> int:

    score = 0

    file_name = Path(path).name.lower()
    path_lower = path.lower()

    if file_name in HIGH_PRIORITY_FILES:
        score += 100

    ARCH_KEYWORDS = [
        "api",
        "route",
        "router",
        "service",
        "agent",
        "controller",
        "core",
        "storage",
        "database",
        "db",
        "graph",
        "llm",
        "retrieval",
        "frontend",
        "backend",
    ]

    for keyword in ARCH_KEYWORDS:
        if keyword in path_lower:
            score += 20

    if "config" in file_name:
        score += 15

    if file_name in NEGATIVE_FILES:
        score -= 100

    return score


def _keep_largest(best_per_file: dict, chunk: dict) -> None:
    """Keep each file's largest chunk; on a tie the one that starts first in the file."""
    current = best_per_file.get(chunk["file_path"])
    if current is None or (len(chunk["text"]), -chunk["start_line"]) > (len(current["text"]), -current["start_line"]):
        best_per_file[chunk["file_path"]] = chunk


def track_largest(chunks: Iterable[dict], best_per_file: dict) -> Iterator[dict]:
    """Pass a chunk stream through unchanged while keeping each file's largest chunk (AD7)."""
    for chunk in chunks:
        _keep_largest(best_per_file, chunk)
        yield chunk


def select_architecture_chunks(chunks: list[dict], max_chunks: int = 12,) -> list[dict]:

    best_per_file = {}

    for chunk in chunks:
        _keep_largest(best_per_file, chunk)

    files = list(best_per_file.values())

    files.sort(
        key=lambda c: score_file(
            c["file_path"]
        ),
        reverse=True,
    )

    return files[:max_chunks]