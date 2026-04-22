#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_ID="${1:-Qwen/Qwen3-8B}"
DEST_DIR="${2:-${PROJECT_ROOT}/models/Qwen3-8B}"

if ! command -v python >/dev/null 2>&1; then
  echo "python is required but was not found in PATH." >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

echo "Downloading ${MODEL_ID} to ${DEST_DIR}"
python - <<'PY' "${MODEL_ID}" "${DEST_DIR}"
import sys
from pathlib import Path

model_id = sys.argv[1]
dest_dir = Path(sys.argv[2]).resolve()

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required. Install it with `python -m pip install huggingface_hub`."
    ) from exc

snapshot_download(
    repo_id=model_id,
    local_dir=str(dest_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"Downloaded {model_id} -> {dest_dir}")
PY

echo "Done."
