#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$project_root/scripts/setup_official_shadowhand.sh"
work_repo="$project_root/external/DexGraspNet_official_compat"
run_name="${1:-shadow_cup_jenga_$(date +%Y%m%d_%H%M%S)}"
batch_size="${2:-16}"
iterations="${3:-6000}"

cd "$work_repo"
exec "$project_root/.venv_l20/bin/python" grasp_generation/main.py \
  --gpu 0 \
  --object_code_list core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03 2_of_Jenga_Classic_Game \
  --scale 0.06 --batch_size "$batch_size" --n_iter "$iterations" --name "$run_name"
