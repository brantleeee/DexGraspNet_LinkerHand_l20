"""Refine generated L20 candidates and compare independent free-object trials.

Run using the mjlab virtual environment. --backend cpu is an explicit oracle;
the default backend uses mjlab.sim.Simulation (MuJoCo-Warp), without PPO.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/l20-mjlab-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/l20-mjlab-mpl")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/l20-mjlab-warp")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from mjlab_grasp.candidates import ROOT, FINGERS, configure_scene, export_reference, load_candidate
from mjlab_grasp.refinement import RefinementConfig, rebuild_contacts, refine, audit_geometry
from mjlab_grasp.validation import GRAVITY, ValidationConfig, validate
from mjlab_grasp.reporting import write_report
from scripts.validate_grasp_dynamics import build_template, project_pose, verify_mesh_identity


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--indices", type=int, nargs="+", help="File indices; default: every saved candidate")
    selection.add_argument("--all", action="store_true", help="Explicitly select every saved candidate (the default)")
    parser.add_argument("--scale", type=float, help="Only test candidates already saved at this scale; never rescale poses")
    parser.add_argument("--execution", choices=["sequential", "batch"], default="sequential",
                        help="Default: finish every case for one candidate before starting the next")
    parser.add_argument("--resume", action="store_true", help="Resume a sequential run; completed candidates are retained")
    parser.add_argument("--dry-run", action="store_true", help="Print candidate selection without simulating or writing files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["mjlab", "cpu"], default="mjlab")
    parser.add_argument("--compare-cpu", action="store_true", help="Also run the identical validation model/controller on CPU")
    parser.add_argument("--cases", choices=list(GRAVITY), nargs="+", default=list(GRAVITY))
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--protocol", choices=["standard", "legacy"], default="standard")
    parser.add_argument("--wrist-motion", choices=["kinematic", "mocap"], default="kinematic",
                        help="Velocity-consistent prescribed wrist; mocap only reproduces historical results")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--refine-seconds", type=float, default=0.5)
    parser.add_argument("--max-joint-delta", type=float, default=0.25)
    parser.add_argument("--max-initial-gap", type=float, default=0.02)
    parser.add_argument("--required-fingers", nargs="+", choices=FINGERS, default=["thumb", "index", "middle"])
    parser.add_argument("--self-collision", action="store_true", help="Enable actual hand self collisions; default retains historical collision masks")
    parser.add_argument("--skip-refinement", action="store_true", help="Validate original candidates only")
    args = parser.parse_args()
    for name in ("friction", "timestep", "hold_seconds", "refine_seconds", "max_joint_delta", "max_initial_gap"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.timestep > 0.002 or args.refine_seconds < args.timestep:
        parser.error("Use timestep <= 0.002 s and at least one refinement step")
    if args.compare_cpu and args.backend != "mjlab":
        parser.error("--compare-cpu requires --backend mjlab")
    for name in ("cases", "required_fingers"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"Duplicate entries in --{name.replace('_', '-')}")
    if "thumb" not in args.required_fingers or len(args.required_fingers) < 2:
        parser.error("Contact template requires thumb and at least one other finger")
    try:
        refinement_config = RefinementConfig(seconds=args.refine_seconds, max_joint_delta=args.max_joint_delta,
                                             max_initial_gap=args.max_initial_gap)
        validation_config = ValidationConfig(protocol=args.protocol, timestep=args.timestep, hold_seconds=args.hold_seconds,
                                             wrist_motion=args.wrist_motion)
    except ValueError as error:
        parser.error(str(error))
    source = args.result_file.resolve()
    from mjlab_grasp.sweep import select_indices, run_sequential

    data = np.load(source, allow_pickle=True)
    try:
        args.indices = select_indices(data, args.indices, args.scale)
    except ValueError as error:
        parser.error(str(error))
    print(f"Source: {source}\nSelected {len(args.indices)} / {len(data)} saved candidates: {args.indices}", flush=True)
    if args.resume and args.execution != "sequential":
        parser.error("--resume requires --execution sequential")
    if args.dry_run:
        print(f"Execution: {args.execution}; backend: {args.backend}; cases: {', '.join(args.cases)}; no simulation started")
        return
    if args.execution == "sequential":
        raise SystemExit(run_sequential(args))
    grasps = [load_candidate(source, i, data) for i in args.indices]
    object_code = grasps[0].get("object_code", source.stem)
    if any(g.get("object_code", source.stem) != object_code for g in grasps):
        parser.error("A batch must refer to one object")
    mesh_file = ROOT / "data/meshdata" / object_code / "coacd/decomposed.obj"
    mesh_digest = hashlib.sha256(mesh_file.read_bytes()).hexdigest()
    if any(g.get("normalized_mesh_sha256", mesh_digest) != mesh_digest for g in grasps):
        parser.error("Object mesh differs from the refined candidate's source mesh")
    manifest = verify_mesh_identity(source, mesh_file)
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    output = args.output.resolve()
    output.mkdir(parents=True)
    save_json(output / "settings.json", {
        "source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "object_mesh_sha256": hashlib.sha256(mesh_file.read_bytes()).hexdigest(),
        "mesh_identity_verified": manifest is not None, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "refinement": asdict(refinement_config), "validation": asdict(validation_config),
        "mujoco_version": mujoco.__version__, "hand_self_collision": args.self_collision,
        "limitations": "Simulation assumptions, not hardware calibration. Fixed PD A/B; no contact-force QP or adaptive controller. Six-direction cases change gravity, not an additional external force.",
    })
    template, robot, _ = build_template(ROOT, output, object_code, args.self_collision)
    asset_files = {Path(m.get("file")).resolve() for m in template.findall("asset/mesh")}
    asset_files.add(ROOT / "grasp_generation/mjcf/linkerhand_l20_right.urdf")
    save_json(output / "asset_manifest.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(asset_files)})
    all_results = []
    batches = {}
    try:
        for index, grasp in zip(args.indices, grasps):
            grasp["object_code"] = object_code
            grasp["normalized_mesh_sha256"] = mesh_digest
            candidate_dir = output / f"candidate_{index:03d}"
            candidate_dir.mkdir()
            projected, feasibility = project_pose(robot, grasp["qpos"])
            grasp["qpos"].update(projected)
            scene = configure_scene(template, grasp, args.friction)
            contacts, mapping = rebuild_contacts(scene, grasp, tuple(args.required_fingers), args.max_initial_gap)
            original_geometry = audit_geometry(scene, grasp, contacts)
            save_json(candidate_dir / "contacts.json", {"contacts": contacts, **mapping})
            save_json(candidate_dir / "source_feasibility.json", feasibility)
            save_json(candidate_dir / "original_geometry.json", original_geometry)
            export_reference(source, index, scene, candidate_dir / "original_reference", grasp)
            variants, labels = [grasp], ["original"]
            valid = [bool(feasibility["source_pose_feasible"] and original_geometry["passed"] and mapping["opposing_object_normals"])]
            if not args.skip_refinement:
                if mapping["within_capture_radius"] and mapping["opposing_object_normals"]:
                    refined, report = refine(scene, grasp, contacts, args.backend, refinement_config, args.timestep)
                    save_json(candidate_dir / "refinement.json", report)
                    if report["termination"] != "trust_region_exceeded":
                        variants.append(refined)
                        labels.append("refined")
                        valid.append(bool(report["passed"] and feasibility["source_pose_feasible"]))
                        np.save(candidate_dir / "refined.npy", [refined], allow_pickle=True)
                        export_reference(source, index, scene, candidate_dir / "refined_reference", refined,
                            {"variant": "refined", "parent_source_index": index, "refinement_report": str(candidate_dir / "refinement.json")})
                else:
                    save_json(candidate_dir / "refinement.json", {"passed": False, "termination": "contact_mapping_rejected", **mapping})
            scale = float(grasp["scale"])
            batch = batches.setdefault(scale, {"scene": scene, "grasps": [], "labels": [], "metadata": []})
            for variant, label, eligible in zip(variants, labels, valid):
                batch["grasps"].append(variant)
                batch["labels"].append(f"candidate_{index:03d}_{label}")
                batch["metadata"].append({"candidate": index, "variant": label,
                    "source_candidate_index": int(grasp.get("source_index", index)),
                    "object_scale": scale, "friction": args.friction, "geometry_and_source_valid": eligible})
            print(f"candidate {index}: prepared {', '.join(labels)}", flush=True)
        # Same object/scale = identical model topology. Each candidate is a
        # separate world with its own wrist mocap, qpos, ctrl and physics state.
        for group_index, batch in enumerate(batches.values()):
            trial_dir = output / f"trials_{group_index:03d}"
            trial_dir.mkdir()
            save_json(trial_dir / "worlds.json", batch["metadata"])
            for backend in ([args.backend, "cpu"] if args.compare_cpu else [args.backend]):
                for case in args.cases:
                    results = validate(batch["scene"], batch["grasps"], batch["labels"], case, trial_dir,
                                       backend, validation_config, tuple(args.required_fingers))
                    for result, metadata in zip(results, batch["metadata"]):
                        result.update(metadata)
                        result["passed"] = bool(result["geometry_and_source_valid"] and result["dynamics_pass"])
                        result["trajectory"] = str((trial_dir / result["trajectory"]).relative_to(output))
                        result["scene"] = str((trial_dir / result["scene"]).relative_to(output))
                        all_results.append(result)
                        print(f"candidate {result['candidate']} {result['variant']} {backend} {case}: pass={result['passed']} slip={result['max_relative_slip_mm']:.2f} mm rotation={result['max_rotation_deg']:.2f} deg", flush=True)
                    save_json(output / "results.json", all_results)
        write_report(output)
        save_json(output / "summary.json", {"completed": True, "trials": len(all_results),
            "passed_trials": sum(r["passed"] for r in all_results),
            "fully_passed": [{"candidate": i, "variant": v, "backend": b} for i in args.indices
                for v in ("original", "refined") for b in ([args.backend, "cpu"] if args.compare_cpu else [args.backend])
                if len(rows := [r for r in all_results if r["candidate"] == i and r["variant"] == v and r["backend"] == b]) == len(args.cases)
                and all(r["passed"] for r in rows)]})
    except BaseException as error:
        save_json(output / "failure.json", {"completed": False, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
