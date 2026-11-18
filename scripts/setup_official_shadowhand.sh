#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_repo="$project_root/external/DexGraspNet_official"
work_repo="$project_root/external/DexGraspNet_official_compat"
commit="bd1b13d7248729af117e1d46aaa6266b147a3c7b"
patch="$project_root/assets/shadowhand_official/generation_compat.patch"
jenga_mesh="$project_root/data/meshdata/2_of_Jenga_Classic_Game"
target_link="$work_repo/data/meshdata/2_of_Jenga_Classic_Game"

if [[ ! -d "$source_repo/.git" ]]; then
  git clone --filter=blob:none https://github.com/PKU-EPIC/DexGraspNet.git "$source_repo"
  git -C "$source_repo" checkout --detach "$commit"
fi
if [[ "$(git -C "$source_repo" rev-parse HEAD)" != "$commit" ]]; then
  echo "Official source checkout differs from pinned commit $commit" >&2
  exit 1
fi
if [[ ! -e "$work_repo/.git" ]]; then
  git -C "$source_repo" worktree add --detach "$work_repo" "$commit"
fi
if [[ "$(git -C "$work_repo" rev-parse HEAD)" != "$commit" ]]; then
  echo "Compatibility worktree differs from pinned commit $commit" >&2
  exit 1
fi
if ! git -C "$work_repo" apply --reverse --check "$patch" >/dev/null 2>&1; then
  git -C "$work_repo" apply --check "$patch"
  git -C "$work_repo" apply "$patch"
fi
if [[ ! -e "$target_link" ]]; then
  ln -s "$jenga_mesh" "$target_link"
fi
if [[ "$(readlink -f "$target_link")" != "$jenga_mesh" ]]; then
  echo "Jenga mesh link points to an unexpected location: $target_link" >&2
  exit 1
fi
echo "Ready: $work_repo ($commit)"
