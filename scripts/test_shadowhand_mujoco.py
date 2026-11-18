"""Validate native ShadowHand cup grasps and optional Jenga transfers/candidates.

No L20 pose conversion, geometry optimization, or RL is applied. The Jenga
transfer mode reuses a pinned official mug pose and is not an official Jenga
grasp. Every selected candidate gets fresh independent dynamics cases.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/shadow-mjlab-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/shadow-mjlab-mpl")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/shadow-mjlab-warp")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from mjlab_grasp.shadowhand import OFFICIAL, verify_official_assets, build_shadow_template, prepare_shadow_grasp
from mjlab_grasp.validation import GRAVITY, ValidationConfig, validate

MUG = "core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03"
JENGA = "2_of_Jenga_Classic_Game"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "mjlab"], default="cpu")
    parser.add_argument("--objects", nargs="+", help="Official object codes; default all five")
    parser.add_argument("--result-file", type=Path, help="Additional native ShadowHand npy for an official object")
    parser.add_argument("--jenga-mode", choices=["none", "transfer", "native"], default="none",
                        help="transfer: reuse official mug ShadowHand pose on local Jenga mesh; native: use --jenga-result-file")
    parser.add_argument("--jenga-result-file", type=Path,
                        help="Native ShadowHand Jenga candidates; L20 joint data are rejected")
    parser.add_argument("--jenga-indices", nargs="+", type=int,
                        help="Native Jenga file indices; default all after scale filtering")
    parser.add_argument("--jenga-mesh", type=Path,
                        default=Path("data/meshdata") / JENGA / "coacd/decomposed.obj")
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--limit", type=int, help="First N per object after scale filtering; default all")
    parser.add_argument("--scale", type=float, help="Filter saved scale; never rescale the grasp")
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--physics-profile", choices=["source", "asset-defaults"], default="source",
                        help="source preserves effective zero damping; asset-defaults activates declared upstream joint physics")
    parser.add_argument("--cases", choices=list(GRAVITY), nargs="+", default=list(GRAVITY))
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        parser.error("duplicate cases would overwrite trajectories")
    if args.limit is not None and args.limit <= 0 or not np.isfinite(args.friction) or args.friction <= 0:
        parser.error("limit/friction must be positive")
    if args.scale is not None and (not np.isfinite(args.scale) or args.scale <= 0):
        parser.error("scale must be positive and finite")
    if args.output.exists():
        parser.error("output already exists; choose a new directory to preserve results")
    if args.jenga_mode == "native" and args.jenga_result_file is None:
        parser.error("--jenga-mode native requires --jenga-result-file")
    if args.jenga_mode != "native" and (args.jenga_result_file is not None or args.jenga_indices is not None):
        parser.error("--jenga-result-file/--jenga-indices require --jenga-mode native")
    if args.jenga_mode != "none" and not args.jenga_mesh.is_file():
        parser.error(f"Jenga mesh is missing: {args.jenga_mesh}")
    source = verify_official_assets()
    files = [args.result_file] if args.result_file else sorted((OFFICIAL / "data/dataset").glob("*.npy"))
    if args.objects:
        files = [p for p in files if p.stem in args.objects]
        if {p.stem for p in files} != set(args.objects):
            parser.error("unknown object code")
    if not files:
        parser.error("no result files found")
    selections = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        indices = args.indices if args.indices is not None else list(range(len(data)))
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(data) for i in indices):
            parser.error(f"invalid/duplicate indices for {path}")
        indices = [i for i in indices if args.scale is None or np.isclose(data[i]["scale"], args.scale, atol=1e-8, rtol=0)]
        if args.limit:
            indices = indices[:args.limit]
        if not indices:
            parser.error(f"no candidates selected for {path}")
        resolved = path.resolve()
        pinned = (resolved.is_relative_to(OFFICIAL.resolve()) and
                  str(resolved.relative_to(OFFICIAL.resolve())) in source["sha256"])
        pose_source = "pinned_official_object_and_pose" if pinned else "user_supplied_native_shadowhand_pose"
        selections.append((path, data, indices, path.stem, None, pose_source))
    if args.jenga_mode == "transfer":
        cups = [s for s in selections if s[3] == MUG and s[5] == "pinned_official_object_and_pose"]
        if len(cups) != 1:
            parser.error("Jenga transfer requires one pinned official mug result file")
        path, data, indices, _, _, _ = cups[0]
        selections.append((path, data, indices, JENGA, args.jenga_mesh,
                           "official_mug_shadowhand_pose_transferred_to_local_jenga_mesh"))
    elif args.jenga_mode == "native":
        path = args.jenga_result_file
        if not path.is_file():
            parser.error(f"Jenga result file is missing: {path}")
        data = np.load(path, allow_pickle=True)
        indices = args.jenga_indices if args.jenga_indices is not None else list(range(len(data)))
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(data) for i in indices):
            parser.error("invalid/duplicate native Jenga indices")
        indices = [i for i in indices if args.scale is None or np.isclose(data[i]["scale"], args.scale, atol=1e-8, rtol=0)]
        if args.limit:
            indices = indices[:args.limit]
        if not indices:
            parser.error("no native Jenga candidates selected")
        selections.append((path, data, indices, JENGA, args.jenga_mesh,
                           "user_supplied_native_shadowhand_pose_local_jenga_mesh"))
    args.output.mkdir(parents=True)
    config = ValidationConfig()
    summary = {"completed": False, "hand": "official DexGraspNet ShadowHand", "source_commit": source["commit"],
               "backend": args.backend, "mujoco_version": mujoco.__version__, "validation_config": asdict(config),
               "friction": args.friction, "cases": args.cases, "refinement": "none; source hand pose unchanged",
               "selected_candidates": sum(len(s[2]) for s in selections), "candidates": [],
               "jenga_mode": args.jenga_mode}
    summary["physics_profile"] = args.physics_profile
    summary["input_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p, _, _, _, _, _ in selections}
    root = Path(__file__).resolve().parents[1]
    summary["code_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                              ("scripts/test_shadowhand_mujoco.py", "mjlab_grasp/shadowhand.py",
                               "mjlab_grasp/physics.py", "mjlab_grasp/validation.py", "mjlab_grasp/candidates.py", "mjlab_grasp/metrics.py")}
    save(args.output / "summary.json", summary)
    for path, data, indices, object_code, object_mesh, pose_source in selections:
        asset_output = args.output / object_code
        template, mechanism = build_shadow_template(object_code, asset_output, args.physics_profile, object_mesh)
        for index in indices:
            grasp = data[index]
            output = asset_output / f"candidate_{index:04d}"
            output.mkdir()
            scene = prepare_shadow_grasp(template, grasp, args.friction)
            entry = {"object": object_code, "source_file": str(path.resolve()), "source_index": index,
                     "scale": float(grasp["scale"]), "pose_source": pose_source,
                     "mechanism": mechanism, "cases": []}
            save(output / "source_grasp.json", {"scale": float(grasp["scale"]),
                "pose_source": pose_source, "qpos": {n: float(v) for n, v in grasp["qpos"].items()}})
            for case in args.cases:
                print(f"{object_code} candidate={index} scale={grasp['scale']:.3f} case={case}", flush=True)
                result = validate(scene, [grasp], ["official"], case, output, args.backend, config)[0]
                entry["cases"].append(result)
                save(output / "results.json", entry)
            entry["all_cases_pass"] = all(r["dynamics_pass"] for r in entry["cases"])
            summary["candidates"].append(entry)
            save(args.output / "summary.json", summary)
    summary["completed"] = True
    summary["passed_candidates"] = sum(e["all_cases_pass"] for e in summary["candidates"])
    summary["passed_trials"] = sum(r["dynamics_pass"] for e in summary["candidates"] for r in e["cases"])
    lines = ["# ShadowHand MuJoCo grasp validation", "", f"Completed: {summary['completed']}; backend: {args.backend}; physics profile: {args.physics_profile}.",
             f"Candidates passing every requested case: {summary['passed_candidates']}/{len(summary['candidates'])}.",
             f"Trials passing strict checks: {summary['passed_trials']}/{sum(len(e['cases']) for e in summary['candidates'])}.",
             "", "Source poses are unchanged. Jenga transfer uses a mug pose when selected; it is not an official Jenga grasp.",
             "22 finger joints, 18 native position actuators, four limited tendons; ideal arm and no object support.",
             "Results reflect this MuJoCo protocol, not the original Isaac Gym validation.", "",
             "| Object | Pose source | Index | Scale | Passed cases |", "|---|---|---:|---:|---:|"]
    for entry in summary["candidates"]:
        lines.append(f"| {entry['object']} | {entry['pose_source']} | {entry['source_index']} | {entry['scale']:.3f} | {sum(r['dynamics_pass'] for r in entry['cases'])}/{len(entry['cases'])} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    save(args.output / "summary.json", summary)
    print(f"Complete: {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
