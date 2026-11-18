"""Exploratory 2x2 template/constraint ablation; does not tune pass thresholds."""

import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjlab_grasp.candidates import ROOT, load_candidate, configure_scene
from mjlab_grasp.refinement import rebuild_contacts, audit_geometry
from mjlab_grasp.validation import validate, ValidationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    code = "sem-Bottle-437678d4bc6be981c8724d5673a063a6"
    grasp = load_candidate(ROOT / f"data/experiments/l20_shape_diagnostic_20261102/results/{code}.npy", 7)
    scene = ET.parse(ROOT / "data/experiments/refinement_shapes_v2_20261102/bottle/trials_000/scene_lift_mjlab.xml").getroot()
    scene.remove(scene.find("sensor"))
    scene = configure_scene(scene, grasp, 3.)
    results = []
    for stiff in [False, True]:
        for fingers in [("thumb", "index", "middle"), ("thumb", "index", "pinky")]:
            configured = copy.deepcopy(scene)
            if stiff:
                for eq in configured.findall("equality/joint"):
                    eq.set("solref", "0.002 1"); eq.set("solimp", "0.99 0.99 0.001")
                for joint in configured.findall("worldbody/.//joint"):
                    joint.set("solreflimit", "0.002 1")
            name = ("stiff" if stiff else "nominal") + "_" + fingers[-1]
            directory = args.output / name
            directory.mkdir()
            contacts, mapping = rebuild_contacts(configured, grasp, fingers)
            geometry = audit_geometry(configured, grasp, contacts)
            row = validate(configured, [grasp], ["original"], "lift", directory, "cpu", ValidationConfig(), fingers)[0]
            row.update(experiment=name, required_fingers=list(fingers), native_geometry=geometry,
                       opposing_target_normals=mapping["opposing_object_normals"],
                       exploratory=True, note="Template selected after observing active fingers; not an unbiased success-rate estimate.")
            results.append(row)
            (args.output/"results.json").write_text(json.dumps(results,indent=2)+"\n")
            print(name, row["dynamics_pass"], row["max_relative_slip_mm"], row["max_mimic_error_rad"], row["final_stable_hold_s"], flush=True)


if __name__ == "__main__":
    main()
