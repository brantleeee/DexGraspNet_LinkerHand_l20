"""Restore a missing example mesh from this repository's bundled Plotly export.

The rank-1 export contains the object vertices at the best grasp's scale.
Undo that scale to recover the normalized mesh expected by ObjectModel.
This does not perform convex decomposition or replace the full mesh dataset.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object-code", default="core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    code = args.object_code
    if Path(code).name != code:
        parser.error("--object-code must be a single directory name")
    source = root / "data/l20_graspdata" / f"{code}_vis_rank1.html"
    destination = root / "data/meshdata" / code / "coacd/decomposed.obj"
    if destination.exists():
        print(f"Already exists: {destination}")
        return

    html = source.read_text()
    start = html.index("[", html.rindex("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html[start:])
    objects = [
        trace for trace in traces
        if trace.get("type") == "mesh3d" and trace.get("color") == "lightgreen"
    ]
    if len(objects) != 1:
        raise ValueError(f"Expected one object mesh, found {len(objects)}")
    # Only load the trusted .npy files bundled with this repository.
    grasps = np.load(source.parent / f"{code}.npy", allow_pickle=True)
    scale = float(min(grasps, key=lambda grasp: grasp["energy"])["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid object scale: {scale}")
    trace = objects[0]
    mesh = trimesh.Trimesh(
        vertices=np.column_stack([trace[axis] for axis in ("x", "y", "z")]) / scale,
        faces=np.column_stack([trace[axis] for axis in ("i", "j", "k")]),
        process=False,
    )
    if not mesh.is_watertight or not np.isfinite(mesh.vertices).all():
        raise ValueError("Exported mesh must be finite and watertight for SDF")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination)
    provenance = {
        "source": str(source.relative_to(root)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scale_removed": scale,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "note": "Recovered from bundled visualization; not a new CoACD decomposition.",
    }
    destination.with_suffix(".source.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Restored {destination}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")


if __name__ == "__main__":
    main()
