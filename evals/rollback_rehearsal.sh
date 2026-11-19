#!/usr/bin/env bash
# evals/rollback_rehearsal.sh — AD11 rollback rehearsal (P10-R).
#
# Builds an index with the NEW code (this checkout's tracked files, working-tree
# versions) on a throwaway local Qdrant in Docker, then reads it with the OLD
# reader (retrieve_chunks at OLD_REF, default 5e48365) from a temporary
# `git worktree`, in its own venv with the old qdrant-client pin. Exit 0 only
# when the old reader returns at least 1 chunk.
#
# Safety: talks only to a local Qdrant. Any QDRANT_URL that is not
# http://localhost:<port> or http://127.0.0.1:<port> is refused. No API key and
# no .env file are read: both runs get a minimal environment, and the new code
# runs from an export that holds tracked files only.
#
# Env (all optional):
#   QDRANT_URL      local URL, default http://127.0.0.1:6399 (the container is published there)
#   OLD_REF         old commit, default 5e48365
#   NEW_PY          python with the new requirements, default python3
#   EMBED_CACHE_DIR new code's baked model cache (required by the new code)
#   OLD_VENV        venv for the old reader, default ~/.cache/codebase-intelligence/rollback-venv-$OLD_REF
#   QDRANT_IMAGE    default qdrant/qdrant:latest
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6399}"
if [[ ! "$QDRANT_URL" =~ ^http://(localhost|127\.0\.0\.1):([0-9]{2,5})/?$ ]]; then
    echo "[rehearsal] REFUSED: QDRANT_URL must be http://localhost:<port> or http://127.0.0.1:<port>, got '$QDRANT_URL'" >&2
    exit 2
fi
PORT="${BASH_REMATCH[2]}"
OLD_REF="${OLD_REF:-5e48365}"
NEW_PY="${NEW_PY:-python3}"
OLD_VENV="${OLD_VENV:-$HOME/.cache/codebase-intelligence/rollback-venv-$OLD_REF}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
: "${EMBED_CACHE_DIR:?set EMBED_CACHE_DIR to the model cache of the new code}"
COLL="rollback-rehearsal"
QUESTION="where are code chunks embedded and stored in Qdrant"

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
WORK="$(mktemp -d)"
NAME="ci-rollback-rehearsal-$$"
cleanup() {
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    git -C "$REPO" worktree remove --force "$WORK/old" >/dev/null 2>&1 || true
    rm -rf "$WORK"
}
trap cleanup EXIT

# Minimal environment for both runs: no secrets, no inherited Qdrant key.
run_env() {
    env -i PATH="$PATH" HOME="$HOME" QDRANT_URL="$QDRANT_URL" QDRANT_API_KEY="" \
        EMBED_CACHE_DIR="$EMBED_CACHE_DIR" PYTHONDONTWRITEBYTECODE=1 "$@"
}

echo "[rehearsal] local Qdrant $QDRANT_IMAGE on 127.0.0.1:$PORT"
docker run -d --rm --name "$NAME" -p "127.0.0.1:$PORT:6333" "$QDRANT_IMAGE" >/dev/null
for _ in $(seq 1 60); do
    curl -fsS "$QDRANT_URL/readyz" >/dev/null 2>&1 && break
    sleep 0.5
done
curl -fsS "$QDRANT_URL/readyz" >/dev/null

echo "[rehearsal] new code: export tracked files of $REPO"
mkdir -p "$WORK/new"
git -C "$REPO" ls-files -z | (cd "$REPO" && tar --null -T - -cf -) | tar -xf - -C "$WORK/new"
(cd "$WORK/new" && run_env "$NEW_PY" -c '
import sys
from core.ingestion.walker import walk_repo
from core.ingestion.chunker import chunk_files
from core.embeddings.embedder import embed_and_store
result = embed_and_store(chunk_files(walk_repo("core")), sys.argv[1])
print(f"[rehearsal] new build: {result}")
sys.exit(0 if result["status"] == "success" and result["chunks_stored"] > 0 else 1)
' "$COLL")

echo "[rehearsal] old reader: worktree at $OLD_REF"
git -C "$REPO" worktree add --detach "$WORK/old" "$OLD_REF" >/dev/null
if [[ ! -x "$OLD_VENV/bin/python" ]]; then
    uv venv --python 3.11 "$OLD_VENV" >/dev/null
fi
# Only the pins the old reader imports, read from the old requirements.txt.
mapfile -t PINS < <(awk '/^(qdrant-client|python-dotenv)/ {print $1}' "$WORK/old/requirements.txt")
uv pip install --quiet --python "$OLD_VENV/bin/python" "${PINS[@]}"
(cd "$WORK/old" && run_env "$OLD_VENV/bin/python" -c '
import sys
from importlib.metadata import version
from core.retrieval.retriever import retrieve_chunks
chunks = retrieve_chunks(sys.argv[2], sys.argv[1])
print("[rehearsal] old reader (qdrant-client %s): %d chunks" % (version("qdrant-client"), len(chunks)))
for c in chunks[:3]:
    print("[rehearsal]   %s:%s-%s %s score=%s" % tuple(c.get(k) for k in ("file_path", "start_line", "end_line", "name", "score")))
ok = bool(chunks) and all(c.get("text") and c.get("file_path") for c in chunks)
sys.exit(0 if ok else 1)
' "$COLL" "$QUESTION")

echo "[rehearsal] PASS: index built by the new code is readable by $OLD_REF"
