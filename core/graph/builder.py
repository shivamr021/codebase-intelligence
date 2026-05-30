"""
core/graph/builder.py — Builds a dependency graph from a repo's import statements.

What this produces:
  A directed graph where:
    - Every node  = a source file in the repo
    - Every edge  = "this file imports that file"

  Example:
    main.py        → imports → database.py
    database.py    → imports → config.py
    auth.py        → imports → database.py
    auth.py        → imports → config.py

  This lets you visually see:
    - Which files are most depended on (high in-degree = core modules)
    - Which files depend on many others (high out-degree = orchestrators)
    - Circular dependencies (A imports B imports A — a real code smell)
    - Entry points (files with no incoming edges)

How it works:
  tree-sitter parses each file's AST.
  We extract import statement nodes specifically.
  We resolve the imported name to an actual file in the repo.
  We add an edge in a NetworkX directed graph.

Why static analysis (not runtime tracing)?
  Runtime tracing would require EXECUTING the code — unsafe, slow, impossible
  for most repos. Static AST analysis reads the source without running it.
  Limitation: dynamic imports (importlib, __import__) are invisible to us.
  We document this limitation honestly — interviewers respect that.

Interview talking point:
  "The dependency graph is built by walking import nodes in the AST of
  each file. I resolve relative imports by mapping the module path to
  actual files in the repo directory. High in-degree nodes in the graph
  represent core shared modules — exactly the files a new developer should
  read first to understand the codebase."
"""

import os
import networkx as nx
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript

# Reuse the same Language objects pattern from chunker.py
PY_LANGUAGE = Language(tspython.language(), "python")
JS_LANGUAGE = Language(tsjavascript.language(), "javascript")
TS_LANGUAGE = Language(tstypescript.language_typescript(), "typescript")

LANGUAGE_MAP = {
    "python":     PY_LANGUAGE,
    "javascript": JS_LANGUAGE,
    "typescript": TS_LANGUAGE,
}


def _get_python_imports(source_bytes: bytes, parser: Parser) -> list[str]:
    """
    Extract imported module names from a Python file's AST.

    Handles two Python import forms:
      import os                    → node type: import_statement
      from pathlib import Path     → node type: import_from_statement

    For "from .auth import User" (relative import):
      The leading dot(s) indicate same-package import.
      We strip the dots and return the module name.

    Returns list of module name strings like:
      ["os", "pathlib", "auth", "database", "config"]
    """
    tree = parser.parse(source_bytes)
    root = tree.root_node
    imports = []

    stack = [root]
    while stack:
        node = stack.pop()

        if node.type == "import_statement":
            # "import os, sys" → children include "dotted_name" nodes
            for child in node.children:
                if child.type in ("dotted_name", "aliased_import"):
                    # Take the first identifier — "os" from "os.path"
                    name = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
                    # Strip alias: "import numpy as np" → "numpy"
                    if " as " in name:
                        name = name.split(" as ")[0].strip()
                    imports.append(name)  # top-level module only

        elif node.type == "import_from_statement":
            # "from auth import User" → we want "auth"
            # Find the module name child
            for child in node.children:
                if child.type == "dotted_name":
                    name = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
                    imports.append(name)
                    break
                elif child.type == "relative_import":
                    # Relative import: "from .auth import ..."
                    # relative_import node contains the module name
                    for subchild in child.children:
                        if subchild.type == "dotted_name":
                            name = source_bytes[subchild.start_byte:subchild.end_byte].decode("utf-8")
                            imports.append(name)
                    break

        stack.extend(node.children)

    return imports


def _get_js_imports(source_bytes: bytes, parser: Parser) -> list[str]:
    """
    Extract imported module paths from a JavaScript/TypeScript file's AST.

    Handles:
      import { foo } from './auth'          → import_statement
      const x = require('./database')       → call_expression with "require"
      import type { MyType } from './types' → import_statement (TS)

    For JS/TS we return the raw path string from the import:
      "./auth", "../config", "express"
    The caller resolves these to actual files.
    """
    tree = parser.parse(source_bytes)
    root = tree.root_node
    imports = []

    stack = [root]
    while stack:
        node = stack.pop()

        if node.type == "import_statement":
            # Find the string node that holds the module path
            for child in node.children:
                if child.type == "string":
                    # Strip quotes: '"./auth"' → './auth'
                    raw = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
                    path = raw.strip('"\'')
                    imports.append(path)
                    break

        elif node.type == "call_expression":
            # require('./database') pattern
            func_node = node.children[0] if node.children else None
            if func_node and source_bytes[func_node.start_byte:func_node.end_byte].decode("utf-8") == "require":
                # Arguments node contains the path string
                for child in node.children:
                    if child.type == "arguments":
                        for arg in child.children:
                            if arg.type == "string":
                                raw = source_bytes[arg.start_byte:arg.end_byte].decode("utf-8")
                                imports.append(raw.strip('"\''))
                        break

        stack.extend(node.children)

    return imports


def _resolve_python_import(
    module_name: str,
    importing_file: str,
    all_files: set[str],
) -> str | None:
    """
    Try to resolve a Python module name to an actual file in the repo.

    Example:
      module_name    = "auth"
      importing_file = "api/routes/ingest.py"
      all_files      = {"core/auth.py", "api/auth.py", "auth.py", ...}

    Strategy:
      1. Look for <module_name>.py anywhere in the repo
      2. Prefer the one closest (same directory first)
      3. If nothing found, it's a third-party/stdlib import — return None

    We return None for unresolved imports — they just don't get an edge.
    This is correct: we don't want edges to "os" or "flask".
    """
    # Direct match: look for module_name.py in all repo files
    module_path = module_name.replace(".", "/")

    normalized_files = {
        f.replace("\\", "/"): f
        for f in all_files
    }

    candidates = [
        original_path
        for normalized_path, original_path in normalized_files.items()
        if normalized_path.endswith(f"{module_path}.py")
    ]

    if not candidates:
        # Also check for package: module_name/__init__.py
        candidates = [
            original_path
            for normalized_path, original_path in normalized_files.items()
            if normalized_path.endswith(
                f"{module_path}/__init__.py"
            )
        ]

    if not candidates:
        return None  # Third-party or stdlib — no edge

    if len(candidates) == 1:
        return candidates[0]

    # Multiple candidates — pick the one in the same directory first
    import_dir = os.path.dirname(importing_file)
    for candidate in candidates:
        if os.path.dirname(candidate) == import_dir:
            return candidate

    # Default to first candidate
    return candidates[0]


def _resolve_js_import(
    import_path: str,
    importing_file: str,
    all_files: set[str],
) -> str | None:
    """
    Resolve a JS/TS import path to an actual file in the repo.

    JS imports are path-based: "./auth", "../config/database"
    We resolve relative to the importing file's directory.

    Third-party imports ("express", "react") don't start with "./" or "../"
    — we skip those immediately.
    """
    # Third-party package — no edge
    if not import_path.startswith("."):
        return None

    import_dir = os.path.dirname(importing_file)
    # Resolve the relative path
    resolved_base = os.path.normpath(os.path.join(import_dir, import_path))
    # Normalise separators to forward slash for consistency
    resolved_base = resolved_base.replace("\\", "/")

    # Try common extensions
    for ext in [".ts", ".tsx", ".js", ".jsx"]:
        candidate = resolved_base + ext
        if candidate in all_files:
            return candidate

    # Try as directory index file
    for ext in [".ts", ".js"]:
        candidate = resolved_base + "/index" + ext
        if candidate in all_files:
            return candidate

    return None


def build_dependency_graph(file_list: list[dict]) -> nx.DiGraph:
    """
    Public interface — builds and returns the full dependency graph.

    Takes the file_list from walker.py (list of file info dicts).
    Returns a NetworkX DiGraph.

    Each node has attributes:
      - language: "python" / "javascript" / "typescript"
      - size: file size in bytes

    Each edge has attributes:
      - weight: 1 (for future use — could count import frequency)

    The graph is used by:
      renderer.py  → to produce the interactive HTML visualisation
      routes/graph.py → served to the frontend
    """
    G = nx.DiGraph()

    # Build a set of relative paths for fast lookup during resolution
    all_rel_paths = {f["rel_path"] for f in file_list}

    # Add all files as nodes first (even isolated ones with no imports)
    for file_info in file_list:
        G.add_node(
            file_info["rel_path"],
            language=file_info["language"],
            size=file_info["size"],
        )

    # Now parse each file and add edges
    for file_info in file_list:
        language = file_info["language"]
        rel_path = file_info["rel_path"]

        # Get the right parser
        lang_obj = LANGUAGE_MAP.get(language)
        if lang_obj is None:
            continue

        parser = Parser()
        parser.set_language(lang_obj)

        # Read the file
        try:
            with open(file_info["path"], "rb") as f:
                source_bytes = f.read()
        except OSError:
            continue

        # Extract raw imports based on language
        if language == "python":
            raw_imports = _get_python_imports(source_bytes, parser)

            for module_name in raw_imports:
                target = _resolve_python_import(
                    module_name,
                    rel_path,
                    all_rel_paths
                )

                if target and target != rel_path:
                    G.add_edge(rel_path, target, weight=1)

        elif language in ("javascript", "typescript"):
            raw_imports = _get_js_imports(source_bytes, parser)
            for import_path in raw_imports:
                target = _resolve_js_import(import_path, rel_path, all_rel_paths)
                if target and target != rel_path:
                    G.add_edge(rel_path, target, weight=1)

    # Log graph stats — useful for debugging and for interviews
    print(
        f"[builder.py] Graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    # Detect and log circular dependencies (cycles in the directed graph)
    cycles = list(nx.simple_cycles(G))
    if cycles:
        print(f"[builder.py] Found {len(cycles)} circular dependencies:")
        for cycle in cycles[:5]:  # Log first 5 only
            print(f"  {' → '.join(cycle)} → {cycle[0]}")
    else:
        print("[builder.py] No circular dependencies found.")

    return G


def get_graph_stats(G: nx.DiGraph) -> dict:
    """
    Compute summary statistics about the dependency graph.
    Returned alongside the graph HTML to the frontend.

    in_degree  = how many files import THIS file (higher = more important)
    out_degree = how many files THIS file imports (higher = more dependencies)
    """
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "cycles": 0, "most_depended_on": []}

    # Most depended-on files (highest in-degree)
    in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    most_depended = [
        {"file": node, "imported_by": count}
        for node, count in in_degrees[:5]
        if count > 0
    ]

    # Count cycles
    cycles = list(nx.simple_cycles(G))

    return {
        "nodes":            G.number_of_nodes(),
        "edges":            G.number_of_edges(),
        "cycles":           len(cycles),
        "most_depended_on": most_depended,
    }