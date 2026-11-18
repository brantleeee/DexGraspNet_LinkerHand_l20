#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$project_root/.venv_isaacgym/bin/python"
python_base="$("$python_bin" -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$python_base/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$project_root/external/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
for argument in "$@"; do
  if [[ "$argument" == "--gui" ]]; then
    # CUDA device 0 does not select the Vulkan renderer on hybrid laptops.
    for driver in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
      if [[ -f "$driver" ]]; then
        export VK_ICD_FILENAMES="$driver"
        export VK_DRIVER_FILES="$driver"
        export __NV_PRIME_RENDER_OFFLOAD=1
        export __GLX_VENDOR_LIBRARY_NAME=nvidia
        echo "Isaac Gym viewer Vulkan driver: $driver"
        break
      fi
    done
  fi
done
cd "$project_root"
exec "$python_bin" scripts/validate_shadowhand_isaacgym.py "$@"
