#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sim_dir=${L20_ISAACSIM_PATH:-/home/lee/isaac-sim}
if [[ ! -x "$sim_dir/python.sh" ]]; then
  echo "Isaac Sim python.sh missing: $sim_dir" >&2
  exit 1
fi
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
  export VK_DRIVER_FILES=$VK_ICD_FILENAMES
fi
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
# One Isaac Sim process at a time: concurrent RTX pipeline compilation can
# make the desktop and GUI unresponsive, even when VRAM is not exhausted.
mkdir -p "$project_dir/data/isaacsim_l20"
exec 9>"$project_dir/data/isaacsim_l20/.runtime.lock"
if ! flock -n 9; then
  echo 'Another L20 Isaac Sim validation is running; queued until it exits.' >&2
  flock 9
fi
output_dir=
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ ${args[i]} == --output ]]; then output_dir=${args[i+1]}; fi
done
"$sim_dir/python.sh" "$project_dir/scripts/evaluate_isaacsim_residual_paired.py" "$@"
# Kit may exit zero after a Python exception; require explicit completion.
if [[ -n "$output_dir" ]]; then
  python3 - "$output_dir" <<'CHECK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);status=json.loads((p/'status.json').read_text())
if status.get('status')!='complete':raise SystemExit(f'RL incomplete; inspect {p}/error.json')
CHECK
fi
