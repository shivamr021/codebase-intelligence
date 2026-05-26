"""
core/ingestion/cloner.py — Clones a public GitHub repo into /tmp/

What this file does:
  1. Takes a GitHub URL string from the API request
  2. Validates it looks like a real GitHub URL
  3. Extracts a clean repo name from the URL (used as folder name + Qdrant collection name)
  4. If the repo was already cloned (exists on disk), deletes it first (fresh clone every time)
  5. Clones it into /tmp/<repo_name>/
  6. Returns the local path so the next step (walker.py) knows where to look

Why fresh clone every time?
  The user might re-submit the same URL after the repo was updated.
  Stale code = wrong analysis. Always fresh = always accurate.
  The performance cost is acceptable for a demo.

Likely failure points (know these for interviews):
  - Private repos: GitPython will hang or throw GitCommandError. We catch this.
  - Invalid URLs: caught by our validation before clone attempt.
  - No internet on Railway: extremely rare, but we catch the generic exception.
  - Repo too large (Linux kernel etc.): clone takes too long. The MAX_FILES limit
    in walker.py handles this at the next step, but the clone itself could still
    be slow. Known limitation.
"""

import os
import shutil
import re
import git  # this is the gitpython library — "import git" not "import gitpython"

from config import TMP_DIR


def extract_repo_name(github_url: str) -> str:
    """
    Pulls a clean repo name out of a GitHub URL.

    Examples:
      "https://github.com/tiangolo/fastapi"        → "fastapi"
      "https://github.com/tiangolo/fastapi.git"    → "fastapi"
      "https://github.com/tiangolo/fastapi/"       → "fastapi"

    How it works:
      - Strip trailing slash and .git suffix
      - Split by "/" and take the last segment
      - That's always the repo name on GitHub URLs
    """
    # Remove trailing slash if present
    url = github_url.rstrip("/")

    # Remove .git suffix if present
    if url.endswith(".git"):
        url = url[:-4]

    # Split by "/" — last part is always the repo name
    repo_name = url.split("/")[-1]

    return repo_name


def validate_github_url(url: str) -> bool:
    """
    Basic check that the URL is a GitHub repo URL.
    Not perfect — it won't catch every edge case — but catches obvious mistakes
    like someone pasting a random URL or leaving the field empty.

    The regex checks for:
      - Starts with http:// or https://
      - Contains github.com
      - Has at least owner/repo in the path (two path segments)
    """
    pattern = r"^https?://github\.com/[\w\-\.]+/[\w\-\.]+/?$"

    # re.match checks from the start of the string
    # The .git suffix is optional so we strip it before matching
    clean_url = url.rstrip("/")
    if clean_url.endswith(".git"):
        clean_url = clean_url[:-4]

    return bool(re.match(pattern, clean_url))


def clone_repo(github_url: str) -> dict:
    """
    Main function — the only one called from outside this file.

    Takes a GitHub URL.
    Returns a dict with:
      {
        "repo_name": "fastapi",          # used as Qdrant collection name
        "local_path": "/tmp/fastapi",    # passed to walker.py
        "status": "cloned"               # or "error"
        "message": "..."                 # human-readable description
      }

    Returning a dict instead of raising exceptions makes it easier for the
    API route to return a clean JSON error response to the frontend.
    """

    # --- Step 1: Validate the URL ---
    if not validate_github_url(github_url):
        return {
            "status": "error",
            "message": (
                f"Invalid GitHub URL: '{github_url}'. "
                "Expected format: https://github.com/owner/repo"
            ),
            "repo_name": None,
            "local_path": None,
        }

    # --- Step 2: Extract repo name ---
    repo_name = extract_repo_name(github_url)

    # --- Step 3: Build the local path where we'll clone to ---
    # os.path.join handles the slash correctly on Windows and Linux
    # Result example: /your-project/tmp/fastapi
    local_path = os.path.join(TMP_DIR, repo_name)

    # --- Step 4: Clean up any previous clone of this repo ---
    # shutil.rmtree removes a directory and everything inside it
    # It's the equivalent of "rm -rf folder/" in bash
    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    # --- Step 5: Ensure the tmp/ directory itself exists ---
    # exist_ok=True means "don't error if it already exists"
    os.makedirs(TMP_DIR, exist_ok=True)

    # --- Step 6: Clone the repo ---
    try:
        # git.Repo.clone_from is the GitPython equivalent of "git clone <url> <path>"
        # depth=1 means "shallow clone" — only download the latest commit, not the
        # entire git history. This is MUCH faster and uses much less disk space.
        # For our purposes (analysing current code) we don't need history.
        git.Repo.clone_from(
            github_url,
            local_path,
            depth=1
        )

        return {
            "status": "cloned",
            "message": f"Successfully cloned '{repo_name}'",
            "repo_name": repo_name,
            "local_path": local_path,
        }

    except git.exc.GitCommandError as e:
        # GitCommandError is thrown when git itself fails:
        # - Repo doesn't exist (404)
        # - Repo is private (authentication required)
        # - Network issue during clone
        # str(e) gives a readable error from git's stderr output
        return {
            "status": "error",
            "message": (
                f"Git clone failed for '{github_url}'. "
                f"Make sure the repo is public and the URL is correct. "
                f"Git error: {str(e)}"
            ),
            "repo_name": None,
            "local_path": None,
        }

    except Exception as e:
        # Catch-all for anything unexpected (disk full, permission error, etc.)
        return {
            "status": "error",
            "message": f"Unexpected error while cloning: {str(e)}",
            "repo_name": None,
            "local_path": None,
        }