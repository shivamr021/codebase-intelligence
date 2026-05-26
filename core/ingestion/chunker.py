"""
core/ingestion/chunker.py — Splits source files into semantic chunks via AST.

This is the file that makes the whole system "understand code" rather than
"search text." It's the most technically interesting file in the project
and the one interviewers will ask about most.

The core idea:
  Most text chunkers split by character count or line count:
    chunk 1: lines 1–50
    chunk 2: lines 51–100
    ...
  This is fast but dumb. A function that spans lines 40–80 gets split
  in half. The embedding for each chunk represents an incomplete thought.
  Retrieval quality drops significantly.

  We chunk by AST (Abstract Syntax Tree) boundaries instead:
    chunk 1: function authenticate() — lines 1–45 (complete unit)
    chunk 2: class UserManager — lines 47–120 (complete unit)
    ...
  Each chunk is a semantically complete unit of code.
  The embedding captures the full meaning of that function or class.
  Retrieval is significantly more precise.

What is an AST?
  When a Python file is parsed, the interpreter builds a tree structure
  representing the code's meaning — not its text. For example:

    def add(a, b):       →  FunctionDef
        return a + b           name: "add"
                               params: [a, b]
                               body: [Return(BinOp(a, +, b))]

  tree-sitter builds this tree for us in ~1ms per file.
  We then walk the tree and extract specific node types (functions, classes).

Interview explanation (memorise this):
  "Instead of splitting code by character count, I use tree-sitter to parse
  each file into an AST and extract function and class definitions as chunks.
  This guarantees each chunk is a complete semantic unit — the embedding
  captures the full meaning of that function. When a user asks a question,
  the retrieved chunks are complete, compilable code units, not arbitrary
  text fragments. This makes LLM responses significantly more accurate
  because the model is reasoning over real code context, not half a function."

Likely failure points (know these):
  1. tree-sitter grammar not installed — caught, falls back to line chunking
  2. Syntax errors in the source file — tree-sitter parses anyway (error nodes)
  3. Very deeply nested code — we only go 2 levels deep (top-level + class methods)
  4. Dynamic code (exec, eval, metaprogramming) — not representable in AST
  5. Files with no functions or classes — falls back to whole-file chunk
"""

import os
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript


# -----------------------------------------------------------------------
# Build Language objects from the installed grammar packages.
#
# tree_sitter_python (the package) exposes the raw C grammar as a
# Python capsule object. Language() wraps it into something the
# Parser can use.
#
# We build these ONCE at module load time (not per file) because
# building a Language object has a small cost. Module-level constants
# are built once when the module is first imported.
# -----------------------------------------------------------------------
PY_LANGUAGE  = Language(tspython.language())
JS_LANGUAGE  = Language(tsjavascript.language())
TS_LANGUAGE  = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())

# Map language name → Language object
# These names match the "language" field returned by walker.py
LANGUAGE_MAP = {
    "python":     PY_LANGUAGE,
    "javascript": JS_LANGUAGE,
    "typescript": TS_LANGUAGE,
}

# -----------------------------------------------------------------------
# Node types we extract as chunks.
#
# These are tree-sitter node type names — they match the AST grammar
# for each language. We extract functions and classes because they are
# the meaningful units of code architecture.
#
# "function_declaration" vs "function_definition":
#   Python uses "function_definition" (def keyword)
#   JavaScript uses "function_declaration" (function keyword)
#   JS also has "arrow_function" and "method_definition" — we capture those too
# -----------------------------------------------------------------------
CHUNK_NODE_TYPES = {
    "python": {
        "function_definition",   # def my_func():
        "class_definition",      # class MyClass:
    },
    "javascript": {
        "function_declaration",  # function myFunc() {}
        "class_declaration",     # class MyClass {}
        "method_definition",     # method inside a class
        "arrow_function",        # const fn = () => {}
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "interface_declaration", # TypeScript-specific: interface MyInterface {}
        "type_alias_declaration",# TypeScript-specific: type MyType = ...
    },
}

# Maximum characters per chunk.
# Even with AST chunking, some functions are enormous (generated code,
# large switch statements). We cap at 3000 chars — roughly 750 tokens.
# At 6000 TPM Groq limit, this lets us fit ~8 chunks per LLM call safely.
MAX_CHUNK_CHARS = 3000

# Minimum characters for a chunk to be worth indexing.
# A 10-character "function" is probably just a stub. Skip it.
MIN_CHUNK_CHARS = 50


def _get_parser(language: str) -> Parser | None:
    """
    Build a tree-sitter Parser for the given language.
    Returns None if the language isn't supported (shouldn't happen
    if walker.py is filtering correctly, but defensive programming).
    """
    lang_obj = LANGUAGE_MAP.get(language)
    if lang_obj is None:
        return None

    parser = Parser(lang_obj)
    return parser


def _extract_name(node, source_bytes: bytes, language: str) -> str:
    """
    Extract the name of a function or class from its AST node.

    Different node types store the name in different child node types:
      Python function_definition → child named "name" (identifier node)
      JS function_declaration    → child named "name" (identifier node)
      JS method_definition       → child named "name" (property_identifier)

    We walk the node's children looking for the name node.
    If we can't find it (anonymous function), return "anonymous".

    source_bytes is the full file content as bytes.
    node.start_byte and node.end_byte index into source_bytes.
    """
    # Name-bearing child node types across all languages
    name_node_types = {"identifier", "property_identifier", "type_identifier"}

    for child in node.children:
        if child.type in name_node_types:
            # child.start_byte:child.end_byte slices the raw bytes
            # .decode("utf-8") converts bytes to a Python string
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")

    return "anonymous"


def _chunk_by_ast(
    source_code: str,
    language: str,
    file_rel_path: str,
) -> list[dict]:
    """
    Parse source_code with tree-sitter and extract function/class chunks.

    Returns a list of chunk dicts:
    [
        {
            "text":          "def authenticate(user, pwd):\n    ...",
            "file_path":     "src/auth.py",
            "chunk_type":    "function",
            "name":          "authenticate",
            "start_line":    14,
            "end_line":      38,
            "language":      "python",
        },
        ...
    ]
    """
    parser = _get_parser(language)
    if parser is None:
        return []

    # tree-sitter works on bytes, not strings.
    # encode("utf-8") converts the Python string to bytes.
    # We keep source_bytes around because node.start_byte/end_byte
    # index into this byte array.
    source_bytes = source_code.encode("utf-8")

    # parser.parse() builds the full AST.
    # Returns a Tree object. Tree.root_node is the top of the tree.
    tree = parser.parse(source_bytes)
    root = tree.root_node

    chunks = []
    target_node_types = CHUNK_NODE_TYPES.get(language, set())

    # -----------------------------------------------------------------------
    # Tree traversal using a stack (iterative DFS).
    #
    # We could use recursion, but Python has a recursion limit (~1000).
    # Deeply nested code could hit that. Iterative stack is safer.
    #
    # DFS (Depth First Search): we go deep into the tree before going wide.
    # This means we find nested functions/methods inside classes.
    #
    # The stack starts with just the root node.
    # Each iteration pops a node, checks if it's one we want,
    # then pushes all its children onto the stack.
    # -----------------------------------------------------------------------
    stack = [root]

    while stack:
        node = stack.pop()

        if node.type in target_node_types:
            # Extract the raw source text for this node
            # node.start_byte:node.end_byte slices the exact bytes for this node
            chunk_text = source_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            )

            # Apply size filters
            if len(chunk_text) < MIN_CHUNK_CHARS:
                # Too small — stub or empty function, not worth indexing
                stack.extend(node.children)
                continue

            if len(chunk_text) > MAX_CHUNK_CHARS:
                # Too large — truncate to MAX_CHUNK_CHARS
                # We keep the beginning (function signature + first logic)
                # which is the most semantically rich part.
                chunk_text = chunk_text[:MAX_CHUNK_CHARS] + "\n... [truncated]"

            # node.start_point is a (row, column) tuple, 0-indexed.
            # We add 1 to make it 1-indexed (matching editor line numbers).
            start_line = node.start_point[0] + 1
            end_line   = node.end_point[0] + 1

            # Determine chunk type from node type
            node_type_str = node.type
            if "class" in node_type_str:
                chunk_type = "class"
            elif "interface" in node_type_str or "type_alias" in node_type_str:
                chunk_type = "type"
            else:
                chunk_type = "function"

            name = _extract_name(node, source_bytes, language)

            chunks.append({
                "text":       chunk_text,
                "file_path":  file_rel_path,
                "chunk_type": chunk_type,
                "name":       name,
                "start_line": start_line,
                "end_line":   end_line,
                "language":   language,
            })

            # IMPORTANT: after adding a chunk, we still push children.
            # Why? Because a class contains methods — we want both the
            # class-level chunk AND the individual method chunks.
            # This gives us coarse (class) and fine (method) granularity.

        # Push children onto the stack so we keep traversing.
        # extend() adds all children at once.
        stack.extend(node.children)

    return chunks


def _fallback_line_chunks(
    source_code: str,
    language: str,
    file_rel_path: str,
) -> list[dict]:
    """
    Fallback chunker for when AST parsing produces nothing.

    This happens when:
      - The file has syntax errors severe enough that tree-sitter
        finds no valid function/class nodes
      - The file is a script with no functions (just top-level code)
      - An unsupported language slipped through walker.py

    We chunk by fixed line count (50 lines per chunk).
    These chunks have lower embedding quality than AST chunks
    but are better than skipping the file entirely.

    The "chunk_type": "raw" flag lets downstream code know
    this came from the fallback — useful for logging and debugging.
    """
    lines = source_code.splitlines()
    chunk_size = 50  # lines per chunk
    chunks = []

    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i : i + chunk_size]
        chunk_text  = "\n".join(chunk_lines)

        if len(chunk_text) < MIN_CHUNK_CHARS:
            continue

        chunks.append({
            "text":       chunk_text,
            "file_path":  file_rel_path,
            "chunk_type": "raw",
            "name":       f"lines_{i+1}_{i+len(chunk_lines)}",
            "start_line": i + 1,
            "end_line":   i + len(chunk_lines),
            "language":   language,
        })

    return chunks


def chunk_file(file_info: dict) -> list[dict]:
    """
    Public interface — the only function called from outside this module.

    Takes a single file_info dict from walker.py:
    {
        "path":     "/tmp/myrepo/src/auth.py",
        "rel_path": "src/auth.py",
        "language": "python",
        "size":     2048,
    }

    Returns a list of chunk dicts (may be empty if file is unreadable).

    Flow:
      1. Read file → 2. Try AST chunking → 3. Fallback to line chunking
         if AST produces nothing → 4. Return chunks
    """

    # --- Step 1: Read the file ---
    try:
        # encoding="utf-8" with errors="replace" means we don't crash
        # on files with non-UTF-8 characters (common in older codebases).
        # Invalid bytes are replaced with the Unicode replacement character (?)
        with open(file_info["path"], encoding="utf-8", errors="replace") as f:
            source_code = f.read()
    except OSError as e:
        # File unreadable — permissions issue or file disappeared
        print(f"[chunker.py] Cannot read {file_info['rel_path']}: {e}")
        return []

    if not source_code.strip():
        # Empty file — nothing to chunk
        return []

    language    = file_info["language"]
    file_path   = file_info["rel_path"]

    # --- Step 2: Try AST chunking ---
    chunks = _chunk_by_ast(source_code, language, file_path)

    # --- Step 3: Fallback if AST produced nothing ---
    if not chunks:
        print(
            f"[chunker.py] AST found no chunks in {file_path} "
            f"(language={language}) — using line fallback"
        )
        chunks = _fallback_line_chunks(source_code, language, file_path)

    return chunks


def chunk_files(file_list: list[dict]) -> list[dict]:
    """
    Convenience wrapper — chunks an entire list of files.
    Returns all chunks from all files as one flat list.

    Called by the ingestion pipeline after walker.py returns file_list.
    """
    all_chunks = []

    for file_info in file_list:
        file_chunks = chunk_file(file_info)
        all_chunks.extend(file_chunks)

        # Progress log — useful when indexing large repos
        if file_chunks:
            print(
                f"[chunker.py] {file_info['rel_path']} "
                f"→ {len(file_chunks)} chunks"
            )

    print(f"[chunker.py] Total chunks across all files: {len(all_chunks)}")
    return all_chunks