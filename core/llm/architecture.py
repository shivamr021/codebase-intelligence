from core.llm.client import call_llm
from core.llm.prompts import (
    EXPLAIN_ARCHITECTURE,
    format_chunks_for_prompt,
)


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


def generate_architecture(chunks: list[dict],) -> tuple[str, str]:

    code_context = format_chunks_for_prompt(chunks)

    files = sorted({
        chunk["file_path"]
        for chunk in chunks
    })

    repo_structure = "\n".join(files)

    prompt = EXPLAIN_ARCHITECTURE.format(
        repo_structure=repo_structure,
        code_context=code_context,
    )

    llm_response = call_llm(
        prompt,
        temperature=0.4,
    )

    if llm_response.startswith("[LLM Error]"):
        return (
            "Architecture generation failed.",
            "flowchart TD\nA[Generation Failed] --> B[Prompt Too Large]"
        )

    summary, mermaid = (parse_architecture_response(llm_response))

    if not mermaid:
        mermaid = (
            "flowchart TD\n"
            "A[Generation Failed] --> "
            "B[Try Re-ingesting]"
        )

    return summary, mermaid