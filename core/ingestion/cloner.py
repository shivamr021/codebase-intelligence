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

from config import TMP_DIR


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


def clone_repo(github_url: str) -> dict:
    """
    Download a public GitHub repo as a zip via GitHub API.
    No git binary required — works on any container.
    """

    if not validate_github_url(github_url):
        return {
            "status": "error",
            "message": f"Invalid GitHub URL: '{github_url}'. Expected: https://github.com/owner/repo",
            "repo_name": None,
            "local_path": None,
        }

    try:
        owner, repo = extract_owner_repo(github_url)
    except Exception:
        return {
            "status": "error",
            "message": f"Could not parse owner/repo from URL: {github_url}",
            "repo_name": None,
            "local_path": None,
        }

    repo_name = repo
    local_path = os.path.join(TMP_DIR, repo_name)

    # Clean previous download
    if os.path.exists(local_path):
        shutil.rmtree(local_path, onerror=_force_remove_readonly)

    os.makedirs(TMP_DIR, exist_ok=True)

    # GitHub zip download URL (downloads default branch)
    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
    fallback_zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"

    print(f"[cloner.py] Downloading repo via GitHub API: {zip_url}")

    zip_path = None
    try:
        # Try main branch first, then master
        response = requests.get(zip_url, timeout=60, allow_redirects=True)
        if response.status_code == 404:
            print(f"[cloner.py] 'main' branch not found, trying 'master'...")
            response = requests.get(fallback_zip_url, timeout=60, allow_redirects=True)

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

        # Extract zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            # GitHub zips have a top-level folder like "repo-main/" or "repo-master/"
            names = zf.namelist()
            top_level = names[0].split("/")[0]  # e.g. "flask-boilerplate-main"

            extract_tmp = os.path.join(TMP_DIR, f"_extract_{repo_name}")
            if os.path.exists(extract_tmp):
                shutil.rmtree(extract_tmp, onerror=_force_remove_readonly)

            zf.extractall(extract_tmp)

        # Move the inner folder to the expected local_path
        inner_path = os.path.join(extract_tmp, top_level)
        shutil.move(inner_path, local_path)
        shutil.rmtree(extract_tmp, onerror=_force_remove_readonly)

        print(f"[cloner.py] ✅ Successfully extracted '{repo_name}' to {local_path}")
        return {
            "status": "cloned",
            "message": f"Successfully downloaded '{repo_name}'",
            "repo_name": repo_name,
            "local_path": local_path,
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