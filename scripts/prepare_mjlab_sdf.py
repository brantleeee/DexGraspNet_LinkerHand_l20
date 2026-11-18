"""Export a separate SDF reference for the existing mjlab grasp task; no simulation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjlab_grasp.sdf import export_sdf_reference

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--object-mesh', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    print(json.dumps(export_sdf_reference(a.reference, a.output, a.object_mesh), indent=2))
