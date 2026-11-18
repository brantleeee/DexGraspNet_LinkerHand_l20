#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mjlab_python="${MJLAB_PYTHON:-/home/lee/robot_learning/mjlab/.venv/bin/python}"
if [[ ! -x "$mjlab_python" ]]; then
    echo "Set MJLAB_PYTHON to the Python executable of your mjlab environment." >&2
    exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/l20-mjlab-uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/l20-mjlab-cache}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec uv run --no-project "$mjlab_python" "$project_root/scripts/run_mjlab_grasp.py" "$@"
