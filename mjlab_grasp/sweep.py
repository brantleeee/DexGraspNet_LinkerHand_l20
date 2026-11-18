"""Sequential, restartable evaluation of every saved candidate.

Each candidate gets a fresh worker process and a new attempt directory. A grasp
failure is a completed evaluation; a worker error is retained for --resume.
"""

import hashlib
import fcntl
import importlib.metadata
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from zipfile import BadZipFile
import xml.etree.ElementTree as ET

import numpy as np

from mjlab_grasp.candidates import ROOT
from mjlab_grasp.reporting import write_report


def select_indices(data, requested=None, scale=None):
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("--scale must be finite and positive")
    indices = list(range(len(data))) if requested is None else list(requested)
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate candidate indices")
    if any(i < 0 or i >= len(data) for i in indices):
        raise ValueError(f"Candidate index outside [0, {len(data)})")
    if scale is not None:
        eligible = [i for i in indices if np.isclose(float(data[i]["scale"]), scale, rtol=1e-6, atol=1e-9)]
        if requested is not None and eligible != indices:
            raise ValueError("An explicitly requested candidate does not match --scale")
        indices = eligible
    if not indices:
        raise ValueError("No saved candidates selected")
    return indices


def save_json(path, value):
    """Replace progress atomically, including when a run is interrupted."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def settings_changes(previous, current, prefix=""):
    """Describe exact changed fingerprints without relaxing run isolation."""
    changes = []
    if isinstance(previous, dict) and isinstance(current, dict):
        for key in sorted(previous.keys() | current.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in previous:
                changes.append({"field": name, "kind": "added", "current": current[key]})
            elif key not in current:
                changes.append({"field": name, "kind": "removed", "previous": previous[key]})
            else:
                changes.extend(settings_changes(previous[key], current[key], name))
    elif previous != current:
        changes.append({"field": prefix, "kind": "modified", "previous": previous, "current": current})
    return changes


def sweep_settings(args):
    options = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k not in {"output", "resume", "dry_run", "all"}}
    source = args.result_file.resolve()
    files = set((ROOT / "mjlab_grasp").glob("*.py"))
    files.update(ROOT / "scripts" / n for n in ("refine_validate_mjlab.py", "validate_grasp_dynamics.py"))
    files.update((ROOT / "grasp_generation/mjcf/meshes").rglob("*"))
    files.add(ROOT / "grasp_generation/mjcf/linkerhand_l20_right.urdf")
    files.update(ROOT / "grasp_generation/mjcf" / name for name in ("contact_points.json", "penetration_points.json"))
    files.update(ROOT / "grasp_generation/utils" / name for name in ("grasp_results.py", "mujoco_compliance.py", "adaptive_compliance.py"))
    manifest_name = source.stem + "_scales.json"
    files.update([source.parent / manifest_name, source.parent.parent.parent / manifest_name])
    data = np.load(source, allow_pickle=True)
    for index in args.indices:
        code = data[index].get("object_code", source.stem)
        files.add(ROOT / "data/meshdata" / code / "coacd/decomposed.obj")
    versions = {}
    for package in ("mujoco", "mjlab", "mujoco-warp", "warp-lang", "torch", "numpy", "scipy", "trimesh"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"schema_version": 2, "options": options,
            "runtime": {"python": sys.version, "executable": str(Path(sys.executable).resolve()), "packages": versions},
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "implementation_and_asset_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(files) if p.is_file()}}


def worker_command(args, index, attempt):
    command = [sys.executable, "-u", str(ROOT / "scripts/refine_validate_mjlab.py"),
               "--execution", "batch", "--result-file", str(args.result_file.resolve()),
               "--indices", str(index), "--output", str(attempt),
               "--backend", args.backend, "--cases", *args.cases,
               "--required-fingers", *args.required_fingers]
    for name in ("friction", "timestep", "protocol", "wrist_motion", "hold_seconds", "refine_seconds", "max_joint_delta", "max_initial_gap"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    for name in ("compare_cpu", "self_collision", "skip_refinement"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    return command


def completed_attempt(candidate_dir, index, cases, backends):
    """Only reuse a complete worker output with the full original case matrix."""
    attempts = [p for p in candidate_dir.glob("attempt_*") if p.is_dir() and re.fullmatch(r"attempt_\d+", p.name)]
    for attempt in sorted(attempts, key=lambda p: int(p.name.split("_")[-1]), reverse=True):
        try:
            summary = json.loads((attempt / "summary.json").read_text())
            rows = json.loads((attempt / "results.json").read_text())
            expected = {(b, c) for b in backends for c in cases}
            original = [(r["backend"], r["case"]) for r in rows if r["variant"] == "original"]
            if not summary.get("completed") or set(original) != expected or len(original) != len(expected):
                continue
            if summary["trials"] != len(rows) or any(r["candidate"] != index for r in rows):
                continue
            if any(type(r["passed"]) is not bool for r in rows) or summary["passed_trials"] != sum(r["passed"] for r in rows):
                continue
            if len({(r["variant"], r["backend"], r["case"]) for r in rows}) != len(rows):
                continue
            variants = {r["variant"] for r in rows}
            if any({(r["backend"], r["case"]) for r in rows if r["variant"] == v} != expected for v in variants):
                continue
            if any(not (attempt / r[key]).is_file() for r in rows for key in ("trajectory", "scene")):
                continue
            if not all(valid_trajectory(attempt / r["trajectory"]) for r in rows):
                continue
            if any(ET.parse(attempt / r["scene"]).getroot().tag != "mujoco" for r in rows):
                continue
            assets = json.loads((attempt / "asset_manifest.json").read_text())
            if not isinstance(assets, dict) or any(not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest
                                                   for p, digest in assets.items()):
                continue
            expected_passed = {(index, v, b) for v in variants for b in backends
                               if all(r["passed"] for r in rows if r["variant"] == v and r["backend"] == b)}
            reported_passed = {(r["candidate"], r["variant"], r["backend"]) for r in summary["fully_passed"]}
            if expected_passed != reported_passed:
                continue
            return attempt, rows, summary
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError):
            continue
    return None


def valid_trajectory(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            time = archive["time"]
            if time.ndim != 1 or len(time) == 0 or not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
                return False
            for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
                array = archive[name]
                initial = archive["initial_" + name]
                if array.shape != (len(time), *initial.shape) or not np.isfinite(array).all() or not np.isfinite(initial).all():
                    return False
                if initial.ndim != (2 if name.startswith("mocap_") else 1):
                    return False
                if name.startswith("mocap_") and initial.shape[-1] != (3 if name == "mocap_pos" else 4):
                    return False
            return True
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return False


def run_sequential(args):
    output = args.output.resolve()
    if output.exists():
        if not args.resume:
            raise ValueError("Output exists; use --resume for this sequential run or choose a new directory")
    else:
        if args.resume:
            raise ValueError("Cannot resume: output directory does not exist")
        output.mkdir(parents=True)
    with (output / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process is using this output directory") from error
        return _run_sequential_locked(args, output)


def _run_sequential_locked(args, output):
    settings = sweep_settings(args)
    if args.resume:
        path = output / "sweep_settings.json"
        if not path.is_file():
            raise ValueError("Cannot resume: sequential settings are missing")
        changes = settings_changes(json.loads(path.read_text()), settings)
        if changes:
            fields = "; ".join(change["field"] for change in changes)
            raise ValueError("Resume rejected: inputs, code, assets, runtime or test options changed; "
                             f"choose a new output directory. Changed: {fields}")
    else:
        save_json(output / "sweep_settings.json", settings)
    backends = [args.backend, "cpu"] if args.compare_cpu else [args.backend]
    states, results, refinement_reports, fully_passed = [], [], [], []
    report_completed = False

    def verify_unchanged():
        changes = settings_changes(settings, sweep_settings(args))
        if changes:
            save_json(output / "settings_changes.json", {"changes": changes})
            if states and states[-1]["status"] == "running":
                states[-1].update(status="error", error="Inputs/code/assets changed during this attempt")
            fields = "; ".join(change["field"] for change in changes)
            raise RuntimeError("Inputs/code/assets changed during evaluation; outputs retained, use a new output directory. "
                               f"Changed: {fields}. Details: {output / 'settings_changes.json'}")

    def persist(finished=False):
        save_json(output / "progress.json", {"total_candidates": len(args.indices), "candidates": states})
        save_json(output / "results.json", results)
        failures = [s for s in states if s["status"] == "error"]
        save_json(output / "errors.json", failures)
        save_json(output / "summary.json", {
            "finished": finished, "completed": report_completed and len(states) == len(args.indices) and all(s["status"] == "completed" for s in states),
            "report_completed": report_completed,
            "total_candidates": len(args.indices), "completed_candidates": sum(s["status"] == "completed" for s in states),
            "error_candidates": len(failures), "trials": len(results),
            "passed_trials": sum(r["passed"] for r in results), "fully_passed": fully_passed})

    try:
        for ordinal, index in enumerate(args.indices, 1):
            verify_unchanged()
            candidate_dir = output / f"candidate_{index:03d}"
            candidate_dir.mkdir(exist_ok=True)
            completed = completed_attempt(candidate_dir, index, args.cases, backends) if args.resume else None
            if completed:
                print(f"[{ordinal}/{len(args.indices)}] candidate {index}: retained completed evaluation", flush=True)
            else:
                numbers = [int(match.group(1)) for p in candidate_dir.glob("attempt_*")
                           if (match := re.fullmatch(r"attempt_(\d+)(?:\.log)?", p.name))]
                attempt_number = max(numbers or [0]) + 1
                attempt = candidate_dir / f"attempt_{attempt_number:04d}"
                log_file = candidate_dir / f"attempt_{attempt_number:04d}.log"
                print(f"[{ordinal}/{len(args.indices)}] candidate {index}: running; log={log_file}", flush=True)
                states.append({"candidate": index, "status": "running", "attempt": str(attempt.relative_to(output)), "log": str(log_file.relative_to(output))})
                persist()
                # The log is flushed by the unbuffered child. A fresh process
                # releases CUDA graphs and model memory after each candidate.
                with log_file.open("w") as log:
                    try:
                        process = subprocess.Popen(worker_command(args, index, attempt), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                        try:
                            returncode = process.wait()
                        except KeyboardInterrupt:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            states[-1]["status"] = "interrupted"
                            raise
                    except OSError as error:
                        log.write(f"Worker launch failed: {error}\n")
                        returncode = 126
                verify_unchanged()
                completed = completed_attempt(candidate_dir, index, args.cases, backends) if returncode == 0 else None
                states.pop()
                if completed is None:
                    states.append({"candidate": index, "status": "error", "returncode": returncode,
                                   "attempt": str(attempt.relative_to(output)), "log": str(log_file.relative_to(output))})
                    print(f"[{ordinal}/{len(args.indices)}] candidate {index}: worker error ({returncode}); continuing", flush=True)
                    persist()
                    continue
            attempt, rows, summary = completed
            states.append({"candidate": index, "status": "completed", "attempt": str(attempt.relative_to(output)),
                           "trials": len(rows), "passed_trials": summary["passed_trials"]})
            for row in rows:
                row = row.copy()
                for key in ("trajectory", "scene"):
                    row[key] = str((attempt / row[key]).relative_to(output))
                results.append(row)
            refinement_reports.extend(attempt.glob("candidate_*/refinement.json"))
            fully_passed.extend(summary["fully_passed"])
            print(f"[{ordinal}/{len(args.indices)}] candidate {index}: completed; {summary['passed_trials']}/{len(rows)} trials passed", flush=True)
            persist()
        persist(finished=True)
        write_report(output, refinement_paths=refinement_reports)
        report_completed = True
        persist(finished=True)
        print(f"Results: {output / 'REPORT.md'}", flush=True)
        return 0 if all(s["status"] == "completed" for s in states) else 1
    except KeyboardInterrupt:
        persist()
        print("Interrupted. Repeat the same command with --resume to continue.", flush=True)
        return 130
    except Exception as error:
        save_json(output / "run_error.json", {"completed": False, "error": str(error)})
        persist()
        raise
