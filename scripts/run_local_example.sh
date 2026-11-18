#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="$project_root/.venv_l20/bin/python"
object_code="core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03"
experiment="${1:-l20_local_$(date +%Y%m%dT%H%M%S)}"
iterations="${2:-6000}"
batch_size="${3:-128}"
if [[ ! "$iterations" =~ ^[1-9][0-9]*$ || ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Iterations and batch size must be positive integers." >&2
    exit 1
fi
if [[ ! "$experiment" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Experiment name may contain only letters, digits, underscores and hyphens." >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

# main.py replaces the logs and results of an existing experiment name.
if [[ -e "data/experiments/$experiment" ]]; then
    echo "Experiment already exists: $experiment. Choose a new name." >&2
    exit 1
fi
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is required"; print("PyTorch", torch.__version__, "GPU", torch.cuda.get_device_name(0))'
"$python_bin" scripts/restore_example_mesh.py --object-code "$object_code"
"$python_bin" grasp_generation/main.py \
    --name "$experiment" --object_code_list "$object_code" \
    --n_contact 4 --n_iter "$iterations" --batch_size "$batch_size"
"$python_bin" grasp_generation/tests/visualize_result.py \
    --object_code "$object_code" \
    --result_path "../data/experiments/$experiment/results"
echo "Result: $project_root/data/experiments/$experiment/results/${object_code}_vis_best.html"
