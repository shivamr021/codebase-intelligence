"""
core/ingestion/cloner.py — Clones a GitHub repo using subprocess.

Why subprocess instead of GitPython:
  GitPython is a wrapper around the git CLI binary. On Railway's container,
  the git binary location is not standard (/usr/bin/git doesn't exist).
  GitPython can't find it and crashes at import time.

  subprocess.run() with shutil.which('git') finds git wherever it is
  on the system PATH — no hardcoded paths, works on any Linux container.
"""

import os
import shutil
import stat
import re
import subprocess

from config import TMP_DIR


def _force_remove_readonly(func, path, _):
    """Windows read-only file handler. No-op on Linux."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _find_git() -> str | None:
    """Find git with more fallbacks"""
    possible = [
        shutil.which("git"),
        "/usr/bin/git",
        "/bin/git",
        "/nix/store/" + "git",   # partial match for nix
    ]
    
    for p in possible:
        if p and os.path.exists(p):
            print(f"[cloner.py] ✅ Found git at: {p}")
            return p
    
    # Last desperate try
    try:
        result = subprocess.run(["which", "git"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            git_path = result.stdout.strip()
            print(f"[cloner.py] ✅ Found git via which: {git_path}")
            return git_path
    except:
        pass

    print("[cloner.py] ❌ git executable not found")
    return None


def extract_repo_name(github_url: str) -> str:
    url = github_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.split("/")[-1]


def validate_github_url(url: str) -> bool:
    pattern = r"^https?://github\.com/[\w\-\.]+/[\w\-\.]+/?$"
    clean_url = url.rstrip("/")
    if clean_url.endswith(".git"):
        clean_url = clean_url[:-4]
    return bool(re.match(pattern, clean_url))


def clone_repo(github_url: str) -> dict:
    """
    Clone a public GitHub repo to /tmp/<repo_name>/.
    Uses subprocess + shutil.which to find git — no GitPython dependency.
    """

    if not validate_github_url(github_url):
        return {
            "status": "error",
            "message": f"Invalid GitHub URL: '{github_url}'. Expected: https://github.com/owner/repo",
            "repo_name": None,
            "local_path": None,
        }

    # Find git binary
    git_path = _find_git()
    if not git_path:
        return {
            "status": "error",
            "message": (
                "git executable not found on this system. "
                "Add 'git' to nixpacks.toml packages."
            ),
            "repo_name": None,
            "local_path": None,
        }

    print(f"[cloner.py] Using git at: {git_path}")

    repo_name = extract_repo_name(github_url)
    local_path = os.path.join(TMP_DIR, repo_name)

    # Clean previous clone
    if os.path.exists(local_path):
        shutil.rmtree(local_path, onerror=_force_remove_readonly)

    os.makedirs(TMP_DIR, exist_ok=True)

    try:
        result = subprocess.run(
            [git_path, "clone", "--depth=1", github_url, local_path],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "message": (
                    f"Git clone failed for '{github_url}'. "
                    f"Make sure the repo is public. "
                    f"Git error: {result.stderr.strip()}"
                ),
                "repo_name": None,
                "local_path": None,
            }

        print(f"[cloner.py] Successfully cloned '{repo_name}' to {local_path}")
        return {
            "status": "cloned",
            "message": f"Successfully cloned '{repo_name}'",
            "repo_name": repo_name,
            "local_path": local_path,
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Clone timed out after 120s. Repository may be too large.",
            "repo_name": None,
            "local_path": None,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error while cloning: {str(e)}",
            "repo_name": None,
            "local_path": None,
        }