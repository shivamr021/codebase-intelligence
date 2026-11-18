"""
core/ingestion/cloner.py — Downloads a GitHub repo via API (no git binary needed).

Why not git CLI:
  Railway's Railpack runtime container is minimal. Git binary reliably available
  only in build containers, not runtime. Instead we use GitHub's zip download API
  which only needs Python's stdlib (urllib/zipfile) + requests (already installed
  as a transitive dep of qdrant-client).
"""

import os
import re
import shutil
import stat
import zipfile
import tempfile

import requests

from config import TMP_DIR, GITHUB_TOKEN


def fetch_head_commit(owner: str, repo: str) -> str | None:
    """
    Fetch the latest commit SHA on the default branch via GitHub API.

    Returns the SHA hex string, or None on any failure.
    When None, the ingestion pipeline gracefully falls back to a full re-index
    (since it can't compare commits without a SHA).

    Rate limits:
      - Unauthenticated: 60 requests/hour
      - With GITHUB_TOKEN: 5,000 requests/hour
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        resp = requests.get(
            url, headers=headers, params={"per_page": 1}, timeout=10
        )
        if resp.status_code == 200:
            commits = resp.json()
            if commits:
                return commits[0]["sha"]
        else:
            print(
                f"[cloner.py] GitHub API returned {resp.status_code} "
                f"for commit lookup — will do full index"
            )
    except Exception as e:
        print(f"[cloner.py] Failed to fetch HEAD commit: {e}")

    return None


def validate_commit_hash(owner: str, repo: str, commit_hash: str) -> str:
    """
    Verify that a commit SHA exists in the given GitHub repo.

    Calls GET /repos/{owner}/{repo}/commits/{sha} — returns 200 if valid,
    422 if the SHA format is wrong, 404 if the commit doesn't exist.

    Returns:
        The full 40-char SHA (GitHub normalizes short SHAs to full).

    Raises:
        ValueError: If the commit hash is invalid or doesn't exist.
        No silent fallbacks — per robust-implementation skill.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_hash}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            full_sha: str = resp.json()["sha"]
            print(f"[cloner.py] Validated commit {full_sha[:12]}...")
            return full_sha
        elif resp.status_code == 404:
            raise ValueError(
                f"Commit '{commit_hash[:12]}...' not found in '{owner}/{repo}'. "
                f"Verify the SHA exists on GitHub."
            )
        elif resp.status_code == 422:
            raise ValueError(
                f"Invalid commit hash format: '{commit_hash}'. "
                f"Must be a valid 40-character hex SHA."
            )
        else:
            raise ValueError(
                f"GitHub API returned HTTP {resp.status_code} when validating "
                f"commit '{commit_hash[:12]}...' in '{owner}/{repo}'."
            )
    except requests.RequestException as e:
        raise ValueError(
            f"Failed to validate commit hash via GitHub API: {e}"
        ) from e


def fetch_branch_head(owner: str, repo: str, branch: str) -> str | None:
    """
    Fetch the HEAD commit SHA for a specific branch via GitHub API.

    Returns the SHA hex string, or None on any failure.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        resp = requests.get(
            url, headers=headers,
            params={"sha": branch, "per_page": 1},
            timeout=10,
        )
        if resp.status_code == 200:
            commits = resp.json()
            if commits:
                return commits[0]["sha"]
        elif resp.status_code == 404:
            raise ValueError(
                f"Branch '{branch}' not found in '{owner}/{repo}'. "
                f"Check the branch name and try again."
            )
        else:
            print(
                f"[cloner.py] GitHub API returned {resp.status_code} "
                f"for branch '{branch}' lookup"
            )
    except ValueError:
        raise  # Re-raise our own ValueError
    except Exception as e:
        print(f"[cloner.py] Failed to fetch branch HEAD: {e}")

    return None


def _force_remove_readonly(func, path, _):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def extract_repo_name(github_url: str) -> str:
    url = github_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.split("/")[-1]


def extract_owner_repo(github_url: str):
    """Returns (owner, repo) tuple from a github URL."""
    url = github_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.rstrip("/").split("/")
    # parts[-1] = repo, parts[-2] = owner
    return parts[-2], parts[-1]


def validate_github_url(url: str) -> bool:
    pattern = r"^https?://github\.com/[\w\-\.]+/[\w\-\.]+/?$"
    clean_url = url.rstrip("/")
    if clean_url.endswith(".git"):
        clean_url = clean_url[:-4]
    return bool(re.match(pattern, clean_url))


def resolve_repo(
    github_url: str,
    branch: str | None = None,
    commit_hash: str | None = None,
) -> dict:
    """
    Phase 1: Validate URL, parse owner/repo, and resolve commit SHA.

    This is CHEAP — no download, just GitHub API calls.
    The caller can use the returned commit_sha to compare against
    stored metadata and abort early if the repo is already up to date.

    Modes:
      - No branch/commit: fetch HEAD of default branch (original behavior)
      - branch="develop": fetch HEAD of that branch
      - commit_hash="abc...": validate that SHA exists on GitHub

    Returns dict with keys: status, message, owner, repo, repo_name, commit_sha, branch.
    """

    if not validate_github_url(github_url):
        return {
            "status": "error",
            "message": f"Invalid GitHub URL: '{github_url}'. Expected: https://github.com/owner/repo",
            "owner": None,
            "repo": None,
            "repo_name": None,
            "commit_sha": None,
            "branch": None,
        }

    try:
        owner, repo = extract_owner_repo(github_url)
    except Exception:
        return {
            "status": "error",
            "message": f"Could not parse owner/repo from URL: {github_url}",
            "owner": None,
            "repo": None,
            "repo_name": None,
            "commit_sha": None,
            "branch": None,
        }

    # Resolve commit SHA based on what the caller provided
    resolved_sha: str | None = None

    if commit_hash:
        # Strict validation — raises ValueError if invalid (no silent fallback)
        resolved_sha = validate_commit_hash(owner, repo, commit_hash)
        print(f"[cloner.py] Validated commit: {resolved_sha[:12]}...")
    elif branch:
        # Fetch HEAD of the specified branch
        resolved_sha = fetch_branch_head(owner, repo, branch)
        print(f"[cloner.py] Branch '{branch}' HEAD: {resolved_sha[:12] if resolved_sha else 'unknown'}...")
    else:
        # Default: fetch HEAD of default branch
        resolved_sha = fetch_head_commit(owner, repo)

    return {
        "status": "resolved",
        "message": f"Resolved '{owner}/{repo}'",
        "owner": owner,
        "repo": repo,
        "repo_name": repo,
        "commit_sha": resolved_sha,
        "branch": branch,
    }


def clone_repo(
    github_url: str,
    owner: str,
    repo: str,
    branch: str | None = None,
    commit_hash: str | None = None,
) -> dict:
    """
    Phase 2: Download the repo as a zip via GitHub API.

    Only called AFTER resolve_repo() and after confirming we actually
    need to download (commit SHA differs from stored).

    Archive URL logic:
      - commit_hash provided: /archive/{sha}.zip  (exact snapshot)
      - branch provided:      /archive/refs/heads/{branch}.zip
      - neither:              /archive/refs/heads/main.zip (fallback to master)

    No git binary required — works on any container.
    """

    repo_name = repo

    # Each run gets its own folder (AD11): two ingests of the same repo name
    # (other branch, other owner) never share or wipe each other's files.
    # The caller deletes work_dir when it is done; on error we delete it here.
    os.makedirs(TMP_DIR, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix=f"{repo_name}-", dir=TMP_DIR)
    cloned = False

    # Build the correct archive URL based on what was requested
    base = f"https://github.com/{owner}/{repo}/archive"

    if commit_hash:
        # Exact commit archive — single URL, no fallback needed
        zip_urls = [f"{base}/{commit_hash}.zip"]
        print(f"[cloner.py] Downloading commit {commit_hash[:12]}...")
    elif branch:
        # Specific branch — single URL
        zip_urls = [f"{base}/refs/heads/{branch}.zip"]
        print(f"[cloner.py] Downloading branch '{branch}'...")
    else:
        # Default: try main, fallback to master
        zip_urls = [
            f"{base}/refs/heads/main.zip",
            f"{base}/refs/heads/master.zip",
        ]
        print(f"[cloner.py] Downloading default branch...")

    zip_path = None
    try:
        # Try each URL in order (single URL for branch/commit, main+master for default)
        response = None
        for url in zip_urls:
            response = requests.get(url, timeout=60, allow_redirects=True)
            if response.status_code == 200:
                break
            print(f"[cloner.py] URL returned {response.status_code}: {url}")

        if response.status_code != 200:
            return {
                "status": "error",
                "message": (
                    f"Could not download repo '{github_url}'. "
                    f"Make sure the repo is public. HTTP {response.status_code}."
                ),
                "repo_name": None,
                "local_path": None,
            }

        # Save zip to temp file
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(response.content)
            zip_path = f.name

        print(f"[cloner.py] Downloaded {len(response.content) / 1024:.1f} KB, extracting...")

        # Extract zip into this run's own folder
        with zipfile.ZipFile(zip_path, "r") as zf:
            # GitHub zips have a top-level folder like "repo-main/" or "repo-master/"
            names = zf.namelist()
            top_level = names[0].split("/")[0]  # e.g. "flask-boilerplate-main"
            zf.extractall(work_dir)

        local_path = os.path.join(work_dir, top_level)
        cloned = True

        print(f"[cloner.py] ✅ Successfully extracted '{repo_name}' to {local_path}")
        return {
            "status": "cloned",
            "message": f"Successfully downloaded '{repo_name}'",
            "repo_name": repo_name,
            "local_path": local_path,
            "work_dir": work_dir,
        }

    except requests.Timeout:
        return {
            "status": "error",
            "message": "Download timed out after 60s. Repository may be too large.",
            "repo_name": None,
            "local_path": None,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error while downloading: {str(e)}",
            "repo_name": None,
            "local_path": None,
        }

    finally:
        if zip_path and os.path.exists(zip_path):
            try:
                os.unlink(zip_path)
            except Exception:
                pass
        if not cloned:
            shutil.rmtree(work_dir, onerror=_force_remove_readonly)