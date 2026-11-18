#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mjlab_python="${MJLAB_PYTHON:-/home/lee/robot_learning/mjlab/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec "$mjlab_python" scripts/refine_validate_mjlab.py "$@"
