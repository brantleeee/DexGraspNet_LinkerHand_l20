"""Evaluate the four newly generated shape families with unchanged dynamics."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mjlab_grasp.candidates import ROOT

OBJECTS = {
    "bottle": "sem-Bottle-437678d4bc6be981c8724d5673a063a6",
    "banana": "ddg-gd_banana_poisson_002",
    "plate": "mujoco-Ecoforms_Plant_Plate_S11Turquoise",
    "camera": "sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/experiments/l20_shape_diagnostic_20261102/results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wrist-motion", choices=["kinematic", "mocap"], default="kinematic")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    study = []
    for label, code in OBJECTS.items():
        source = args.source / f"{code}.npy"
        data = np.load(source, allow_pickle=True)
        target = args.output / label
        command = [sys.executable, "-u", str(ROOT / "scripts/refine_validate_mjlab.py"),
            "--result-file", str(source), "--all", "--execution", "batch", "--cases", "gz-", "lift",
            "--wrist-motion", args.wrist_motion, "--output", str(target)]
        print(f"{label}: evaluating {len(data)} generated candidates", flush=True)
        with (args.output / f"{label}.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        row = {"object": label, "object_code": code, "source": str(source), "candidates": len(data),
            "generation_iterations": 3000, "generation_seed": 1, "wrist_motion": args.wrist_motion,
            "scale": [float(x) for x in np.unique([g['scale'] for g in data])],
            "generation_geometry_pass_count": sum(bool(g['E_fc'] < .3 and g['E_dis'] < .005 and g['E_pen'] < .001) for g in data),
            "generation_energy_min": float(min(g['energy'] for g in data)), "returncode": process.returncode}
        if (target / "summary.json").exists():
            row["evaluation"] = json.loads((target / "summary.json").read_text())
        study.append(row)
        (args.output / "study.json").write_text(json.dumps(study, indent=2) + "\n")
        print(label, row, flush=True)


if __name__ == "__main__":
    main()
