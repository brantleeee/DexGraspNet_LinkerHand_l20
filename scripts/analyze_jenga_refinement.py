"""Offline geometry/phase diagnosis of original HTML candidates 2 and 12.

Optional controlled CPU ablations vary only equality/limit settings or timestep.
Original experiment artifacts and production physics settings are never changed.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/l20-diagnosis-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear
import trimesh

from mjlab_grasp.candidates import ROOT, initialize_data, load_candidate
from mjlab_grasp.physics import compile_scene, independent_transmission
from mjlab_grasp.validation import validate, ValidationConfig


def decode(value):
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
    return np.array(value)


def html_meshes(path):
    source = path.read_text()
    start = source.find("[", source.rfind("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(source[start:])
    return [(t.get("name", "object"), np.column_stack([decode(t[a]) for a in "xyz"]),
             np.column_stack([decode(t[a]) for a in "ijk"]).astype(int))
            for t in traces if t.get("type") == "mesh3d"]


def collision_mesh(model, data, geom):
    mid = model.geom_dataid[geom]
    vs, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fs, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    vertices = model.mesh_vert[vs:vs+vn] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    return trimesh.Trimesh(vertices, model.mesh_face[fs:fs+fn], process=False)


def diagnose_contacts(scene, grasp, contacts, refinement):
    model = compile_scene(scene)
    data = mujoco.MjData(model)
    data.qpos[:] = initialize_data(model, [grasp])[0][0]
    mujoco.mj_forward(model, data)
    oid = model.body("object").id
    object_meshes = [collision_mesh(model, data, g) for g in range(model.ngeom)
                     if model.geom_bodyid[g] == oid and model.geom_contype[g]]
    object_planes = [ConvexHull(m.vertices).equations for m in object_meshes]
    transmission, qadr, _ = independent_transmission(model)
    joints = model.actuator_trnid[:, 0]
    bounds = model.actuator_ctrlrange
    lower = np.maximum(bounds[:, 0] - data.qpos[qadr], -0.25)
    upper = np.minimum(bounds[:, 1] - data.qpos[qadr], 0.25)
    original = data.qpos.copy()
    results = []
    for contact in contacts:
        bid = model.body(contact["body"]).id
        point = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ contact["point_local"]
        hand_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_contype[g]]
        hand = trimesh.util.concatenate([collision_mesh(model, data, g) for g in hand_geoms])
        _, pad_distance, _ = trimesh.proximity.closest_point_naive(hand, point[None])
        equations = ConvexHull(hand.vertices).equations
        inside_hand = np.max(equations[:, :3] @ point + equations[:, 3]) < -1e-6
        target = np.array(contact["target_world"])
        internal_parts = [i for i, eq in enumerate(object_planes) if np.max(eq[:, :3] @ target + eq[:, 3]) < -1e-5]
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, point, bid)
        reduced = jac @ transmission
        fit = lsq_linear(reduced, target - point, bounds=(lower - 1e-8, upper + 1e-8), tol=1e-10)
        row = {**contact, "initial_point_world": point.tolist(),
            "pad_collision_surface_distance_mm": float(pad_distance[0]*1000), "pad_inside_collision_hull": bool(inside_hand),
            "target_inside_other_collision_parts": internal_parts,
            "linearized_target_residual_mm": float(np.linalg.norm(reduced @ fit.x - (target-point))*1000),
            "linearized_drive_delta_rad": {model.joint(j).name: float(v) for j, v in zip(joints, fit.x)}}
        results.append(row)
    if "raw_joint_pos" in refinement:
        for name, value in refinement["raw_joint_pos"].items():
            jid = model.joint(name).id
            data.qpos[model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(model, data)
        for row in results:
            bid = model.body(row["body"]).id
            point = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ row["point_local"]
            row["final_target_error_mm"] = float(np.linalg.norm(point - row["target_world"])*1000)
    data.qpos[:] = original
    return results


def phase_diagnosis(trajectory):
    with np.load(trajectory) as a:
        t = a["time"]
        thresholds = {"slip_m": 0.005, "rotation_deg": 15., "penetration_m": 0.001,
                      "mimic_error_rad": 0.01, "joint_limit_error_rad": 0.001}
        onset = {k: float(t[ids[0]]) if len(ids := np.flatnonzero(a[k] > v)) else None for k, v in thresholds.items()}
        phases = []
        for name, lo, hi in [("zero_gravity", 0., .3), ("gravity_ramp", .3, .5),
                             ("prelift_hold", .5, 1.), ("lift", 1., 2.), ("postlift_hold", 2., 4.01)]:
            mask = (t >= lo) & (t < hi)
            if not mask.any():
                continue
            phases.append({"phase": name, **{f"max_{k}": float(a[k][mask].max()) for k in thresholds},
                "mean_finger_normal_force_N": a["finger_normal_force_N"][mask].mean(0).tolist(),
                "last_finger_normal_force_N": a["finger_normal_force_N"][mask][-1].tolist()})
        return {"first_threshold_crossing_s": onset, "phases": phases}


def self_proximity_decomposition(scene, grasp):
    """Decompose the generator's point-distance E_spen, not mesh penetration."""
    model = compile_scene(scene)
    data = mujoco.MjData(model)
    data.qpos[:] = initialize_data(model, [grasp])[0][0]
    mujoco.mj_forward(model, data)
    groups = json.loads((ROOT / "grasp_generation/mjcf/penetration_points.json").read_text())
    points, bodies, names = [], [], []
    for name, local in groups.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0 or not local:
            continue
        local = np.array(local)
        points.extend(local @ data.xmat[bid].reshape(3,3).T + data.xpos[bid])
        bodies.extend([bid] * len(local)); names.extend([name] * len(local))
    points, bodies = np.array(points), np.array(bodies)
    distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
    penalty = np.maximum(0., .02 - distance)
    penalty[distance < 1e-6] = 0
    same = bodies[:, None] == bodies[None]
    adjacent = ((model.body_parentid[bodies][:, None] == bodies[None]) |
                (bodies[:, None] == model.body_parentid[bodies][None])) & ~same
    return {"total_point_proximity_sum_m": float(penalty.sum()),
            "same_link_constant_sum_m": float(penalty[same].sum()),
            "parent_child_sum_m": float(penalty[adjacent].sum()),
            "remaining_pairs_sum_m": float(penalty[~same & ~adjacent].sum()),
            "source_E_spen": float(grasp["E_spen"]),
            "note": "The same-link contribution is rigid-transform invariant and has no useful pose gradient; this sum is not maximum physical self-penetration."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "data/experiments/mjlab_jenga_all_manual_20261101")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate the figure in an existing diagnosis directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.plot_only)
    source = ROOT / "data/experiments/l20_jenga_006_screen_20261030/results/2_of_Jenga_Classic_Game.npy"
    rows = json.loads((args.run / "results.json").read_text())
    diagnosis = {}
    figure = plt.figure(figsize=(15, 10))
    for row_number, index in enumerate([2, 12]):
        candidate = args.run / f"candidate_{index:03d}/attempt_0001/candidate_{index:03d}"
        contacts = json.loads((candidate / "contacts.json").read_text())["contacts"]
        refinement = json.loads((candidate / "refinement.json").read_text())
        original_row = next(r for r in rows if r["candidate"] == index and r["variant"] == "original" and r["case"] == "lift")
        scene = ET.parse(args.run / original_row["scene"]).getroot()
        # Saved evaluation XML has sensors; compilation adds its own sensors.
        sensor = scene.find("sensor")
        if sensor is not None:
            scene.remove(sensor)
        grasp = load_candidate(source, index)
        geometry = diagnose_contacts(scene, grasp, contacts, refinement)
        phase = {r["variant"]: phase_diagnosis(args.run / r["trajectory"])
                 for r in rows if r["candidate"] == index and r["case"] == "lift"}
        diagnosis[index] = {"contacts": geometry, "trajectory_phases": phase,
            "generation_energy": {k: float(grasp[k]) for k in grasp if k == "energy" or k.startswith("E_")}}
        html_path = source.parent.parent / "visualizations_generated" / f"2_of_Jenga_Classic_Game_vis_candidate_{index:03d}.html"
        meshes = html_meshes(html_path)
        vertices = np.concatenate([m[1] for m in meshes])
        center = (vertices.min(0) + vertices.max(0))/2
        radius = np.ptp(vertices, axis=0).max()/2 * 1.05
        for view, azimuth in enumerate([-45, 65, 170]):
            ax = figure.add_subplot(2, 3, row_number*3 + view+1, projection="3d")
            for name, v, f in meshes:
                color = "#d6aa45" if name == "object" else ("#d98683" if name.startswith("thumb") else "#72adcc")
                ax.add_collection3d(Poly3DCollection(v[f], facecolor=color, linewidths=0, alpha=1., rasterized=True))
            for c, color in zip(geometry, ["red", "orange", "purple"]):
                p, target = np.array(c["initial_point_world"]), np.array(c["target_world"])
                ax.scatter(*p, color=color, s=20)
                ax.plot(*np.stack([p, target]).T, color=color, linewidth=2)
            ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                   zlim=(center[2]-radius, center[2]+radius), title=f"Original HTML candidate {index} / azimuth {azimuth}")
            ax.set_box_aspect([1,1,1]); ax.view_init(elev=15, azim=azimuth)
            ax.set_axis_off()
        if args.ablate and not args.plot_only:
            ablations = []
            for name, stiff, dt in [("historical", False, .001), ("stiff_coupling", True, .001), ("stiff_coupling_dt025ms", True, .00025)]:
                configured = ET.fromstring(ET.tostring(scene))
                if stiff:
                    for eq in configured.findall("equality/joint"):
                        eq.set("solref", "0.002 1"); eq.set("solimp", "0.99 0.99 0.001")
                    for joint in configured.findall("worldbody/.//joint"):
                        joint.set("solreflimit", "0.002 1")
                directory = args.output / f"candidate_{index:03d}_{name}"
                directory.mkdir()
                for case in ["gz-", "lift"]:
                    result = validate(configured, [grasp], ["original"], case, directory, "cpu", ValidationConfig(timestep=dt, wrist_motion="mocap"))
                    ablations.extend({**r, "ablation": name, "candidate": index} for r in result)
                print(f"candidate {index} ablation {name} complete", flush=True)
            diagnosis[index]["ablations"] = ablations
        if not args.plot_only:
            (args.output / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    figure.tight_layout()
    figure.savefig(args.output / "original_html_views.png", dpi=150)
    plt.close(figure)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
