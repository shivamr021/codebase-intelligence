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

from config import GROQ_API_KEY

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

# Model we use for all generation tasks.
# llama-3.3-70b-versatile: 70B params, 128K context, strong at code reasoning.
# Free tier on Groq. Verify this model ID at console.groq.com if it errors.
MODEL = "llama-3.3-70b-versatile"

# Max tokens to generate per response.
# 2048 is enough for architecture explanations and bug reports.
# Raising this uses more of your free quota per call.
MAX_TOKENS = 2048


def call_llm(prompt: str, temperature: float = 0.3) -> str:
    """
    Send a prompt to Groq, return the response text.

    Args:
        prompt      : The full prompt string (already has code context injected).
        temperature : 0.0 = deterministic, 1.0 = creative.
                      0.3 is good for code analysis — consistent but not robotic.

    Returns:
        The model's response as a plain string.
        On any unrecoverable error, returns an error message string
        (never raises) so the API route can still return a clean JSON response.

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
        # Strip removes leading/trailing whitespace from the response.
        return response.choices[0].message.content.strip()

    except RateLimitError:
        # 429 — hit the rate limit. Wait and retry once.
        # This happens if multiple requests fire in quick succession.
        print("[client.py] Rate limit hit — waiting 10s before retry...")
        time.sleep(10)

    except APIError as e:
        # Non-rate-limit API error (wrong model name, server error, etc.)
        # Return immediately — retrying won't help.
        return f"[LLM Error] Groq API error: {str(e)}"

    except Exception as e:
        # Catch-all — network failure, timeout, etc.
        return f"[LLM Error] Unexpected error: {str(e)}"

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
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Both attempts failed. Return a clean error string.
        return (
            f"[LLM Error] Failed after retry. "
            f"You may have hit Groq's free tier rate limit. "
            f"Wait 1 minute and try again. Details: {str(e)}"
        )