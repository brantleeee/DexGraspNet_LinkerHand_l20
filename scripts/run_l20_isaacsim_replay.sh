#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sim_dir=${L20_ISAACSIM_PATH:-/home/lee/isaac-sim}
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=$VK_ICD_FILENAMES
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
mkdir -p "$project_dir/data/isaacsim_l20"
exec 9>"$project_dir/data/isaacsim_l20/.runtime.lock"
if ! flock -n 9; then
  echo 'Another L20 Isaac Sim instance is running; queued until it exits.' >&2
  flock 9
fi
args=("$@")
status_file=
for ((i=0; i<${#args[@]}; i++)); do
  if [[ ${args[i]} == --status-file ]]; then status_file=${args[i+1]}; fi
done
if [[ -z "$status_file" ]]; then
  status_file="$project_dir/data/isaacsim_l20/replay_$(date +%Y%m%d_%H%M%S)_$$.json"
  args+=(--status-file "$status_file")
fi
"$sim_dir/python.sh" "$project_dir/scripts/replay_l20_isaacsim.py" "${args[@]}"
python3 - "$status_file" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
status=json.loads(p.read_text())
if status.get('status')!='closed':raise SystemExit(f'Replay did not close normally; inspect {p}')
PY
