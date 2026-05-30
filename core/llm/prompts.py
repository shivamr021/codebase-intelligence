"""
core/llm/prompts.py — All prompt templates in one place.

Why one file for all prompts?
  When a prompt gives bad output, you fix it here.
  When you want to improve response quality, you experiment here.
  No hunting across 5 different route files.

Template syntax:
  Each prompt uses Python's .format() placeholders: {variable_name}
  Call it like: ANSWER_QUESTION.format(code_context=chunks, question=q)

Design principle for each prompt:
  1. Tell the model its role clearly
  2. Give it the code context (retrieved chunks)
  3. Give it the specific task
  4. Tell it the output format you expect
  5. Tell it to cite file names and line numbers

The "cite file and line numbers" instruction is critical.
Without it, the model makes up plausible-sounding but wrong locations.
With it, you get answers like "in src/auth.py at line 34" which are
verifiable and make the system feel trustworthy.
"""

# -----------------------------------------------------------------------
# ANSWER_QUESTION
# Used by: api/routes/query.py
# Task: Answer a natural language question about the codebase
# -----------------------------------------------------------------------
ANSWER_QUESTION = """You are an expert software engineer analysing a codebase.

Below are the most relevant code chunks retrieved from the repository,
along with their file paths and line numbers:

{code_context}

---

Based ONLY on the code above, answer this question:
{question}

Rules:
- Cite the exact file path and line number for every claim you make.
  Example: "The authentication logic is in src/auth.py at line 34."
- If the code context does not contain enough information to answer,
  say "I cannot find enough information in the indexed code to answer this."
  Do NOT guess or hallucinate.
- Be concise and technical. The person asking is a developer.
"""


# -----------------------------------------------------------------------
# FIND_BUGS
# Used by: api/routes/bugs.py
# Task: Identify potential bugs and code quality issues
# Output: Structured JSON so the frontend can render a table
# -----------------------------------------------------------------------
FIND_BUGS = """You are an expert code reviewer analysing a codebase for bugs.

Below are code chunks from the repository:

{code_context}

---

Identify potential bugs, code smells, and issues in the code above.

Return your response as a JSON array with this exact structure:
[
  {{
    "file": "src/auth.py",
    "line": 34,
    "severity": "high",
    "issue": "One sentence description of the problem",
    "suggestion": "One sentence fix or improvement"
  }}
]

Severity levels: "high" (crash/security risk), "medium" (logic error), "low" (code smell)

Rules:
- Only report real issues visible in the provided code. Do not invent issues.
- Maximum 10 issues. Prioritise by severity.
- Return ONLY the JSON array. No explanation text before or after it.
- Escape any double quotes inside strings with backslash.
"""


# -----------------------------------------------------------------------
# EXPLAIN_ARCHITECTURE
# Used by: api/routes/diagram.py
# Task: Describe the codebase architecture and return a Mermaid diagram
# Output: Mermaid flowchart syntax (rendered by frontend with mermaid.js)
# -----------------------------------------------------------------------
# EXPLAIN_ARCHITECTURE = """You are a software architect analysing a codebase.

# Below are code chunks from the repository, selected to represent the
# overall architecture:

# {code_context}

# ---

# Perform two tasks:

# TASK 1 — Architecture Summary:
# Write 3-5 sentences explaining the overall architecture of this codebase.
# What does it do? What are the main components? How do they interact?

# TASK 2 — Mermaid Diagram:
# Generate a Mermaid flowchart showing the main components and their relationships.
# Use this exact format:

# ```mermaid
# graph TD
#     A[ComponentName] --> B[AnotherComponent]
#     B --> C[ThirdComponent]
# ```

# Rules:
# - Use real file/module names from the code context, not generic labels.
# - Maximum 12 nodes in the diagram — keep it readable.
# - Include the full mermaid code block with the triple backticks.
# - Architecture summary FIRST, then the mermaid diagram.
# """

EXPLAIN_ARCHITECTURE = """You are a senior software architect producing a codebase map.
 
Given the code chunks below, output TWO things:
 
1. SUMMARY (2–3 sentences, plain English): what this codebase does, its main tech, and its key entry points.
 
2. MERMAID DIAGRAM: a high-level architectural flowchart of the repository.
 
DIAGRAM RULES — follow exactly or the renderer will break:

- Start the block with: ```mermaid
- First line inside block: flowchart TD
- Node IDs: alphanumeric and underscores only
- Node labels containing spaces must use quotes
- Every node used in an edge must be defined
- Use arrows like: A --> B or A -->|"calls"| B
- Use subgraphs for architectural layers

ARCHITECTURE LIMITS:

- Show only the most important architectural components
- Maximum 15 nodes
- Maximum 25 edges
- Focus on architecture, not individual files
- Prefer modules, services, routes, models, databases, and external systems
- Combine related implementation files into a single architectural node
- Do NOT attempt to represent every file in the repository

VALID SUBGRAPHS:

subgraph ENTRY["Entry Points"]
subgraph API["API / Routes"]
subgraph CORE["Core Logic"]
subgraph DATA["Data / Models"]
subgraph CONFIG["Config"]
subgraph UTILS["Utilities"]

- No backticks inside node labels
- End the Mermaid block with ```

The Mermaid diagram MUST be syntactically valid Mermaid.
Do not output explanatory text inside the Mermaid block.
Do not reference nodes that are not defined.
 
RESPOND IN EXACTLY THIS FORMAT — nothing else:
 
SUMMARY: <your 2-3 sentence description>
 
```mermaid
flowchart TD
<your diagram here>
```
 
Code chunks:
{code_context}
"""

# -----------------------------------------------------------------------
# SUGGEST_REFACTOR
# Used by: api/routes/query.py (when question is about improvements)
# Task: Suggest specific refactoring improvements
# -----------------------------------------------------------------------
SUGGEST_REFACTOR = """You are a senior software engineer reviewing code for improvement.

Below are code chunks from the repository:

{code_context}

---

Suggest concrete refactoring improvements for this code.

For each suggestion:
1. Name the file and function/class to improve (with line number)
2. Explain the current problem in one sentence
3. Describe the improvement in 2-3 sentences
4. If possible, show a short before/after code snippet

Focus on:
- Reducing code duplication
- Improving error handling
- Simplifying complex logic
- Better naming and readability
- Performance improvements that are obvious from the code

Maximum 5 suggestions. Prioritise impact.
"""


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Formats a list of chunk dicts into a readable string for injection
    into any of the prompt templates above.

    Input: list of chunk dicts from retriever.py
    Output: formatted string like:

      --- File: src/auth.py | Function: authenticate | Lines: 14-38 ---
      def authenticate(user, pwd):
          ...

      --- File: src/models.py | Class: User | Lines: 5-45 ---
      class User:
          ...

    This format makes it easy for the LLM to cite specific locations.
    """
    if not chunks:
        return "No code context available."

    formatted_parts = []

    for chunk in chunks:
        header = (
            f"--- File: {chunk['file_path']} | "
            f"{chunk['chunk_type'].capitalize()}: {chunk['name']} | "
            f"Lines: {chunk['start_line']}-{chunk['end_line']} ---"
        )
        formatted_parts.append(f"{header}\n{chunk['text']}")

    # Join with double newline so each chunk is visually separated
    return "\n\n".join(formatted_parts)