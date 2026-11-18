"""Export an indexed grasp as an mjlab reference, without running simulation."""

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjlab_grasp.candidates import ROOT, configure_scene, export_reference, load_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = ROOT / "data/experiments/l20_small_20261030"
    parser.add_argument("--result-file", type=Path, default=source / "results/scales/scale_0p06/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03.npy")
    parser.add_argument("--candidate-index", type=int, default=7)
    parser.add_argument("--dynamics-dir", type=Path, default=source / "dynamics_scale_0p06")
    parser.add_argument("--scene", type=Path, help="Explicit scene XML; overrides the legacy dynamics-dir filename")
    parser.add_argument("--output", type=Path, default=ROOT / "data/mjlab_grasp/scale_0p06_candidate_7")
    args = parser.parse_args()
    grasp = load_candidate(args.result_file, args.candidate_index)
    scene_file = args.scene or args.dynamics_dir / f"candidate_{args.candidate_index:03d}_mu3_gz-_adaptive.xml"
    scene = ET.parse(scene_file).getroot()
    # Set all pose/scale fields from this candidate, including the wrist.
    # Existing XML supplies collision assets, actuator properties and friction.
    friction = float(scene.find(".//geom[@friction]").get("friction").split()[0])
    scene = configure_scene(scene, grasp, friction)
    export_reference(args.result_file, args.candidate_index, scene, args.output, grasp)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
