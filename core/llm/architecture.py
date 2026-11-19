import re

from core.llm.client import call_llm
from core.llm.prompts import (
    EXPLAIN_ARCHITECTURE,
    format_chunks_for_prompt,
)

# Summary returned when the architecture step fails; callers must not save it.
ARCHITECTURE_FAILED = "Architecture generation failed."


_RESERVED_IDS = ("graph", "end", "subgraph", "flowchart")
_LABEL_OR_ID = re.compile(
    r'(\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|"[^"]*"|\|[^|]*\|)'
    r"|(?<![\w.-])(" + "|".join(_RESERVED_IDS) + r")(?![\w.-])"
)


def rename_reserved_ids(mermaid: str) -> str:
    """Rename Mermaid keywords used as node ids (e.g. `graph`) so it parses."""
    out = []
    for i, line in enumerate(mermaid.split("\n")):
        if i == 0 or line.strip() == "end":
            out.append(line)
            continue
        head = ""
        if line.strip().startswith("subgraph"):
            idx = line.index("subgraph") + len("subgraph")
            head, line = line[:idx], line[idx:]
        out.append(head + _LABEL_OR_ID.sub(
            lambda m: m.group(1) or m.group(2) + "_node", line))
    return "\n".join(out)


def parse_architecture_response(llm_response: str,) -> tuple[str, str]:

    mermaid_start = llm_response.find("```mermaid")

    mermaid_end = llm_response.find("```",mermaid_start + 10,)

    if mermaid_start == -1:
        return llm_response.strip(), ""

    summary = llm_response[:mermaid_start].strip()

    mermaid_raw = llm_response[mermaid_start : mermaid_end + 3]

    mermaid_lines = mermaid_raw.split("\n")

    mermaid = "\n".join(mermaid_lines[1:-1]).strip()

    return summary, mermaid


def generate_architecture(
    chunks: list[dict],
    architecture_context: dict,
    file_paths: list[str] | None = None,
) -> tuple[str, str]:
    """file_paths: every walked file (AD7); without it the names come from chunks.
    chunks: the code samples sent; callers pass only changed files' chunks on an incremental run."""

    code_context = (
        "Use dependency graph information first. "
        "Code samples are secondary.\n\n"
        + format_chunks_for_prompt(chunks)
    )
    files = sorted(set(file_paths) if file_paths is not None else {
        chunk["file_path"]
        for chunk in chunks
    })

    repo_structure = "\n".join(files)

    prompt = EXPLAIN_ARCHITECTURE.format(
        node_count=architecture_context["node_count"],
        edge_count=architecture_context["edge_count"],
        entry_points="\n".join(
            architecture_context["entry_points"]
        ),
        most_depended_on="\n".join(
            architecture_context["most_depended_on"]
        ),
        largest_modules="\n".join(
            architecture_context["largest_modules"]
        ),
        dependencies="\n".join(
            architecture_context["dependencies"]
        ),
        repo_structure=repo_structure,
        code_context=code_context,
    )

    llm_response = call_llm(
        prompt,
        temperature=0.4,
    )

    # Any LLM error (including an empty reply) or a reply with no mermaid
    # block is a failure: return the sentinel so nothing gets saved and the
    # next ingest retries, instead of storing a fake "Generation Failed" chart.
    if llm_response.startswith("[LLM Error]"):
        print(f"[architecture.py] {llm_response[:300]}")
        return ARCHITECTURE_FAILED, ""

    summary, mermaid = (parse_architecture_response(llm_response))

    if not mermaid:
        print("[architecture.py] LLM reply has no mermaid block.")
        return ARCHITECTURE_FAILED, ""

    return summary, rename_reserved_ids(mermaid)