"""
core/llm/client.py — Groq API wrapper.

One job: take a prompt string, return a response string.
Everything else in the project calls THIS function.
Nobody else touches the Groq SDK directly.

Why centralise it here?
  If Groq changes their API, or we switch models, or we add retry logic —
  we change ONE file. Not 5 different files scattered across the project.

Why not hardcode the API key here?
  The key is a SECRET. If you push it to GitHub even once, bots scan
  public repos within minutes and can abuse your quota. We load it
  from config.py which reads it from .env (local) or Railway Variables (prod).
  .env is in .gitignore so it is never committed.
"""

import time
from openai import OpenAI, RateLimitError, APIError

from config import GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_MODEL

# -----------------------------------------------------------------------
# OpenAI client pointed at Groq's endpoint.
#
# Groq is OpenAI-API-compatible — same SDK, same method calls.
# The ONLY differences from a normal OpenAI setup are:
#   1. api_key  → your Groq key (not OpenAI key)
#   2. base_url → Groq's server (not OpenAI's server)
#
# This is why we pip installed 'openai' and not a 'groq' package.
# The openai SDK is just an HTTP client — point it anywhere.
# -----------------------------------------------------------------------
_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Model and token limit come from config.py (env GROQ_MODEL, GROQ_MAX_TOKENS).
# llama-3.3-70b-versatile was shut down; gpt-oss-120b is a reasoning model
# and needs 8192 tokens, or architecture replies come back empty.
MODEL = GROQ_MODEL
MAX_TOKENS = GROQ_MAX_TOKENS


LLM_ERROR_PREFIX = "[LLM Error]"


def _reply(response) -> tuple[str, bool]:
    """
    (reply text, cut_off), or an [LLM Error] when the model sent nothing.
    Reasoning models can spend every token on hidden reasoning and send
    "" or None (finish_reason=length); that is a failure, not an answer.
    A reply with text and finish_reason=length is real but cut off.
    """
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        return f"{LLM_ERROR_PREFIX} Empty reply from the model (finish_reason={choice.finish_reason})", False
    return text, choice.finish_reason == "length"


def call_llm(prompt: str, temperature: float = 0.3) -> str:
    """Send a prompt to Groq, return the response text (see call_llm_reply)."""
    return call_llm_reply(prompt, temperature)[0]


def call_llm_reply(prompt: str, temperature: float = 0.3) -> tuple[str, bool]:
    """
    Send a prompt to Groq, return (response text, cut_off).
    cut_off is True when the model stopped at the token limit with text.

    Args:
        prompt      : The full prompt string (already has code context injected).
        temperature : 0.0 = deterministic, 1.0 = creative.
                      0.3 is good for code analysis — consistent but not robotic.

    Returns:
        (text, cut_off). On any unrecoverable error, text is an error message
        starting with "[LLM Error]" and cut_off is False (never raises), so
        each route decides how to report it (/ask answers 502).

    Retry logic:
        Groq free tier has rate limits (30 RPM, 6000 TPM).
        If we hit a 429 (RateLimitError), we wait 10 seconds and try once more.
        If it fails again, we return an error string — don't crash the server.
    """

    # --- Attempt 1 ---
    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                # System message sets the model's role.
                # Keeping it short — the real instructions are in prompts.py.
                {
                    "role": "system",
                    "content": (
                        "You are an expert software engineer and code analyst. "
                        "Be precise, technical, and cite specific file names and "
                        "line numbers when referencing code."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=temperature,
        )

        # response.choices[0].message.content is where Groq puts the text.
        return _reply(response)

    except RateLimitError:
        # 429 — hit the rate limit. Wait and retry once.
        # This happens if multiple requests fire in quick succession.
        print("[client.py] Rate limit hit — waiting 10s before retry...")
        time.sleep(10)

    except APIError as e:
        # Non-rate-limit API error (wrong model name, server error, etc.)
        # Return immediately — retrying won't help.
        return f"{LLM_ERROR_PREFIX} Groq API error: {str(e)}", False

    except Exception as e:
        # Catch-all — network failure, timeout, etc.
        return f"{LLM_ERROR_PREFIX} Unexpected error: {str(e)}", False

    # --- Attempt 2 (only reached after RateLimitError + sleep) ---
    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert software engineer and code analyst. "
                        "Be precise, technical, and cite specific file names and "
                        "line numbers when referencing code."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=temperature,
        )
        return _reply(response)

    except Exception as e:
        # Both attempts failed. Return a clean error string.
        return (
            f"{LLM_ERROR_PREFIX} Failed after retry. "
            f"You may have hit Groq's free tier rate limit. "
            f"Wait 1 minute and try again. Details: {str(e)}"
        ), False