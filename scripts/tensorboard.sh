#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tensorboard_python="$project_root/.venv_l20/bin/python"

if [[ ! -x "$tensorboard_python" ]]; then
    echo "Missing $tensorboard_python; create the project environment first." >&2
    exit 1
fi

exec "$tensorboard_python" -m tensorboard.main \
    --logdir "$project_root/data/experiments" \
    --host 127.0.0.1 \
    --port 6006 \
    "$@"
