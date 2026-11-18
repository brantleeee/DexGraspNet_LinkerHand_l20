"""Small CPU contact ablations of the two specified Jenga candidates."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjlab_grasp.candidates import ROOT, load_candidate
from mjlab_grasp.refinement import refine, RefinementConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = ROOT / "data/experiments/mjlab_jenga_all_manual_20261101"
    rows = json.loads((run / "results.json").read_text())
    source = ROOT / "data/experiments/l20_jenga_006_screen_20261030/results/2_of_Jenga_Classic_Game.npy"
    results = []
    for index in [2, 12]:
        candidate = run / f"candidate_{index:03d}/attempt_0001/candidate_{index:03d}"
        contacts = json.loads((candidate / "contacts.json").read_text())["contacts"]
        row = next(r for r in rows if r["candidate"] == index and r["variant"] == "original" and r["case"] == "lift")
        scene = ET.parse(run / row["scene"]).getroot()
        scene.remove(scene.find("sensor"))
        original = load_candidate(source, index)
        base = RefinementConfig()
        configurations = [("historical", base),
            ("attraction_off_margin1mm", replace(base, stiffness=0.)),
            ("attraction_off_margin0", replace(base, stiffness=0., margin=0.)),
            ("attraction_on_margin0", replace(base, margin=0.)),
            ("attraction_on_margin025mm", replace(base, margin=0.00025)),
            ("maxforce05N_only", replace(base, max_force=0.5))]
        for name, config in configurations:
            grasp, report = refine(scene, original, contacts, "cpu", config)
            report["legacy_geometry_acceptance"] = all(v for k, v in report.get("quality_checks", {}).items() if k != "target_convergence")
            report.update(candidate=index, experiment=name)
            results.append(report)
            print(index, name, report["termination"], "delta", report["max_joint_excursion_rad"],
                  "error_mm", report["final_target_error_m"] * 1000, "passed", report["passed"], flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
