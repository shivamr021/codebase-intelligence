"""
core/ingestion/walker.py — Walks a cloned repo and returns source files.

What it does:
  Given a local path (from cloner.py), recursively find every source code
  file worth analysing. Filter out noise (tests, build artifacts, generated
  files). Return a clean list of file paths + their language.

Why filtering matters:
  A real repo has hundreds of files that add zero value to code analysis:
    - node_modules/ (thousands of dependency files)
    - __pycache__/ (compiled Python bytecode)
    - .git/ (version control internals)
    - test files (we want to analyse the actual logic, not the tests)
    - minified JS (.min.js) — unreadable by any human or LLM
    - lock files (package-lock.json, poetry.lock) — generated, not authored

  Feeding these into the LLM wastes tokens, degrades analysis quality,
  and could push us over Groq's TPM limit mid-request.

Interview talking point:
  "I filter by AST-parseable file types and exclude generated/dependency
  directories before chunking. This keeps the token count predictable
  and ensures every chunk represents intentionally authored code."
"""

import os
from config import MAX_FILES

# -----------------------------------------------------------------------
# File extensions we support.
# These map to languages tree-sitter can parse.
# Key: extension (with dot), Value: language name string.
# The language name is passed to tree-sitter's parser loader in chunker.py.
# -----------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".py":  "python",
    ".js":  "javascript",
    ".ts":  "typescript",
    ".jsx": "javascript",   # JSX uses the javascript grammar
    ".tsx": "typescript",   # TSX uses the typescript grammar
}

# -----------------------------------------------------------------------
# Directories to skip entirely.
# os.walk lets us modify the dirs list in-place to skip subtrees.
# This is important — without it, walking node_modules alone could
# return 50,000+ files.
# -----------------------------------------------------------------------
SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    ".git",
    ".github",
    "venv",
    "env",
    ".venv",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    ".pytest_cache",
    "migrations",       # Django/Alembic DB migrations — generated code
    "static",           # Static assets
    "assets",
    "public",
    "vendor",
}

# -----------------------------------------------------------------------
# File name patterns to skip.
# These are common generated/config files that add no analytical value.
# Using endswith() checks — faster than regex for simple suffix matching.
# -----------------------------------------------------------------------
SKIP_SUFFIXES = (
    ".min.js",      # Minified JavaScript — unreadable
    ".min.css",     # Minified CSS
    ".lock",        # Lock files (package-lock.json, poetry.lock)
    ".map",         # Source maps
    "_pb2.py",      # Protobuf generated files
    ".generated.ts",
)

# Files to skip by exact name
SKIP_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "setup.py",         # Often boilerplate
    "conftest.py",      # Pytest config
}


def is_test_file(filename: str) -> bool:
    """
    Returns True if a file looks like a test file.
    We skip tests because they test the logic, not implement it.
    The LLM analysing test files produces lower-quality architecture insights.

    Patterns caught:
      test_auth.py, auth_test.py, auth.test.js, auth.spec.ts
    """
    name = filename.lower()
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.js")
        or name.endswith(".spec.ts")
    )


def walk_repo(local_path: str) -> list[dict]:
    """
    Main function — called from the ingestion pipeline.

    Takes the local repo path returned by cloner.py.
    Returns a list of file info dicts:
    [
        {
            "path":     "/tmp/myrepo/src/auth.py",   # absolute path
            "rel_path": "src/auth.py",               # relative to repo root
            "language": "python",                    # for tree-sitter
            "size":     2048,                        # bytes — used for logging
        },
        ...
    ]

    Returns empty list (not an error) if no supported files are found.
    The MAX_FILES cap is enforced here — first N files found, then stop.
    """

    found_files = []

    # os.walk() generates (dirpath, dirnames, filenames) for every
    # directory in the tree, starting from local_path.
    #
    # The key trick: modifying `dirs` IN PLACE with [:] = filtered list
    # tells os.walk to not descend into those directories.
    # This is the standard Python idiom for pruning a walk — important
    # to know because it's not obvious from the docs.
    for dirpath, dirs, filenames in os.walk(local_path):

        # Prune skip directories in-place.
        # dirs[:] modifies the list that os.walk is iterating over.
        # Without [:], we'd be reassigning the local variable, not the list.
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:

            # Check file count cap first — stop if we've hit MAX_FILES
            if len(found_files) >= MAX_FILES:
                print(
                    f"[walker.py] Reached MAX_FILES limit ({MAX_FILES}). "
                    f"Stopping walk. Increase MAX_FILES env var to index more."
                )
                return found_files

            # Skip by exact filename
            if filename in SKIP_FILENAMES:
                continue

            # Skip test files
            if is_test_file(filename):
                continue

            # Skip by suffix patterns (minified, generated, etc.)
            if any(filename.endswith(s) for s in SKIP_SUFFIXES):
                continue

            # Check if the extension is one we support
            _, ext = os.path.splitext(filename)
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Build the full absolute path
            full_path = os.path.join(dirpath, filename)

            # Build the relative path (relative to repo root).
            # This is what we store in Qdrant metadata and show in the UI.
            # os.path.relpath computes the relative path between two absolutes.
            rel_path = os.path.relpath(full_path, local_path)

            # Get file size in bytes for logging
            # os.path.getsize returns bytes — we log this to spot
            # suspiciously large files that might cause token overflow.
            try:
                size = os.path.getsize(full_path)
            except OSError:
                # File disappeared between walk and stat (rare, but possible)
                continue

            # Skip empty files — nothing to analyse
            if size == 0:
                continue

            # Skip very large files (over 100KB).
            # A 100KB Python file is ~3000+ lines. Rare in practice.
            # More importantly, a single file this size could overflow
            # the context window when combined with other chunks.
            if size > 50_000:
                print(f"[walker.py] Skipping large file: {rel_path} ({size} bytes)")
                continue

            found_files.append({
                "path":     full_path,
                "rel_path": rel_path,
                "language": SUPPORTED_EXTENSIONS[ext],
                "size":     size,
            })

    print(f"[walker.py] Found {len(found_files)} source files in {local_path}")
    return found_files