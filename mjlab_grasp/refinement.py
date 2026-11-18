"""Fixed-target virtual contact forces for local L20 pose refinement."""

import copy
from dataclasses import dataclass, asdict
import json

import mujoco
import numpy as np
import torch
import trimesh

from mjlab_grasp.candidates import ROOT, initialize_data
from mjlab_grasp.physics import Physics, compile_scene, independent_transmission, point_jacobian


@dataclass
class RefinementConfig:
    seconds: float = 0.5
    stiffness: float = 100.0
    damping: float = 0.03
    posture_stiffness: float = 0.2
    max_force: float = 2.0
    max_joint_delta: float = 0.25
    max_initial_gap: float = 0.02
    contact_tolerance: float = 0.002
    margin: float = 0.001
    max_penetration: float = 0.001

    def __post_init__(self):
        for name, value in asdict(self).items():
            allow_zero = name in ("stiffness", "damping", "posture_stiffness", "margin", "max_penetration")
            if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f"Invalid refinement parameter: {name}={value}")


def rebuild_contacts(scene, grasp, required_fingers=("thumb", "index", "middle"), max_gap=0.02):
    """Choose one nearby annotated pad point per required finger, then freeze targets.

    Nearest points use the ACTUAL compiled object collision meshes, including mesh
    centering/rotation and the candidate scale. No convex internal-face SDF signs.
    """
    model = compile_scene(scene)
    qpos, _ = initialize_data(model, [grasp])
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    oid = model.body("object").id
    meshes = []
    for geom in np.flatnonzero(model.geom_bodyid == oid):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        mid = model.geom_dataid[geom]
        vs, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fs, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        vertices = model.mesh_vert[vs:vs+vn] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        meshes.append(trimesh.Trimesh(vertices, model.mesh_face[fs:fs+fn], process=False))
    mesh = trimesh.util.concatenate(meshes)
    pads = json.loads((ROOT / "grasp_generation/mjcf/contact_points.json").read_text())
    contacts = []
    for finger in required_fingers:
        choices = []
        for name, local in pads.items():
            if not name.startswith(finger + "_"):
                continue
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            local = np.array(local)
            world = local @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
            target, distance, triangles = trimesh.proximity.closest_point_naive(mesh, world)
            k = int(distance.argmin())
            choices.append((distance[k], name, local[k], target[k], mesh.face_normals[triangles[k]]))
        if not choices:
            raise ValueError(f"No contact pad candidates for {finger}")
        distance, name, local, target, normal = min(choices, key=lambda x: x[0])
        contacts.append({"finger": finger, "body": name, "point_local": local.tolist(),
                         "target_world": target.tolist(), "normal_world": normal.tolist(),
                         "initial_gap_m": float(distance)})
    thumb = next((c for c in contacts if c["finger"] == "thumb"), None)
    opposing = thumb is not None and any(
        np.dot(thumb["normal_world"], c["normal_world"]) < -0.2
        for c in contacts if c["finger"] != "thumb")
    return contacts, {"contact_source": "reconstructed_from_pad_candidates",
        "within_capture_radius": all(c["initial_gap_m"] <= max_gap for c in contacts),
        "opposing_object_normals": bool(opposing)}


def audit_geometry(scene, grasp, contacts, tolerance=0.002, max_penetration=0.001):
    model = compile_scene(scene)
    states, _ = initialize_data(model, [grasp])
    data = mujoco.MjData(model)
    data.qpos[:] = states[0]
    mujoco.mj_forward(model, data)
    oid = model.body("object").id
    object_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == oid and model.geom_contype[g]]
    gaps = {}
    point_errors = {}
    for contact in contacts:
        bid = model.body(contact["body"]).id
        hand_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_contype[g]]
        distances = [mujoco.mj_geomDistance(model, data, h, o, 1.0, None) for h in hand_geoms for o in object_geoms]
        gaps[contact["body"]] = float(min(distances, default=1.0))
        point = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ contact["point_local"]
        point_errors[contact["body"]] = float(np.linalg.norm(point - contact["target_world"]))
    depth = max([max(0., -c.dist) for c in data.contact if oid in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])] or [0.])
    passed = depth <= max_penetration and all(g <= tolerance for g in gaps.values())
    return {"passed": bool(passed), "required_link_gap_m": gaps, "max_penetration_m": float(depth),
            "contact_point_error_m": point_errors, "max_contact_point_error_m": max(point_errors.values(), default=0.)}


def refine(scene, grasp, contacts, backend="mjlab", config=None, timestep=0.001):
    config = config or RefinementConfig()
    if not np.isfinite(timestep) or timestep <= 0 or config.seconds < timestep or not contacts:
        raise ValueError("Refinement requires contacts and at least one positive simulation step")
    model = compile_scene(scene, fixed_object=True, torque_control=True, margin=config.margin, torque_damping=config.damping)
    physics = Physics(model, [grasp], backend, timestep)
    matrix, qadr, vadr = independent_transmission(model)
    finger_dofs = model.jnt_dofadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
    ids = [model.body(c["body"]).id for c in contacts]
    local = [c["point_local"] for c in contacts]
    with physics.stream_context(), torch.no_grad():
        transmission = physics.tensor(matrix)
        target = physics.tensor([c["target_world"] for c in contacts])[None]
        original = physics.read("qpos")[:, qadr].clone()
        effort = physics.tensor(model.actuator_forcerange)
        bounds = physics.tensor([list(map(float, a.get("ctrlrange").split())) for a in scene.find("actuator")])
        lower, upper = bounds[:, 0], bounds[:, 1]
        trace = []
        settled = 0.0
        reason = "iteration_limit"
        excursion = 0.0
        for step in range(round(config.seconds / timestep)):
            points, jac = point_jacobian(physics, ids, local)
            error = target - points
            force = config.stiffness * error
            force *= (config.max_force / force.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1)
            reduced = jac @ transmission
            torque = torch.einsum("bcju,bcj->bu", reduced, force)
            q = physics.read("qpos")[:, qadr]
            torque += config.posture_stiffness * (original - q)
            # Restore toward the local trust region before exceeding it.
            lo = torch.maximum(lower, original - config.max_joint_delta)
            hi = torch.minimum(upper, original + config.max_joint_delta)
            torque += 5 * (torch.maximum(lo - q, torch.zeros_like(q)) - torch.maximum(q - hi, torch.zeros_like(q)))
            torque = torch.maximum(effort[:, 0], torch.minimum(effort[:, 1], torque))
            physics.write("ctrl", torque)
            physics.step()
            delta = (physics.read("qpos")[:, qadr] - original).abs().max().item()
            excursion = max(excursion, delta)
            gap = error.norm(dim=-1).max().item()
            speed = physics.read("qvel")[:, finger_dofs].abs().max().item()
            settled = settled + timestep if gap <= config.contact_tolerance and speed < 0.05 else 0.0
            if step % 10 == 0:
                trace.append([step * timestep, gap, speed, delta])
            if delta > config.max_joint_delta + 0.01:
                reason = "trust_region_exceeded"
                break
            if settled >= 0.025:
                reason = "converged"
                break
        state = physics.read("qpos")[0].cpu().numpy()
        points, _ = point_jacobian(physics, ids, local)
        final_target_error = (target - points).norm(dim=-1).max().item()
        final_speed = physics.read("qvel")[:, vadr].abs().max().item()
        final_finger_speed = physics.read("qvel")[:, finger_dofs].abs().max().item()
    refined = copy.deepcopy(grasp)
    energy_keys = [k for k in refined if k == "energy" or k.startswith("E_")]
    refined["parent_optimization_metrics"] = {k: refined.pop(k) for k in energy_keys}
    refined["variant"] = "refined"
    raw_joint_pos = {}
    for jid in range(model.njnt):
        name = model.joint(jid).name
        raw_joint_pos[name] = float(state[model.jnt_qposadr[jid]])
        refined["qpos"][name] = raw_joint_pos[name]
    drive_projection = 0.0
    for a, jid in enumerate(model.actuator_trnid[:, 0]):
        name = model.joint(jid).name
        value = float(np.clip(refined["qpos"][name], lower[a].item(), upper[a].item()))
        drive_projection = max(drive_projection, abs(value - refined["qpos"][name]))
        refined["qpos"][name] = value
    # Export exact mimic state; equality integration has finite compliance.
    projection_error = 0.0
    for e in range(model.neq):
        child, parent = model.eq_obj1id[e], model.eq_obj2id[e]
        cn, pn = model.joint(child).name, model.joint(parent).name
        value = model.eq_data[e, 0] + model.eq_data[e, 1] * refined["qpos"][pn]
        projection_error = max(projection_error, abs(refined["qpos"][cn] - value))
        refined["qpos"][cn] = float(value)
    audit = audit_geometry(scene, refined, contacts, config.contact_tolerance, config.max_penetration)
    legal = all(lower[i].item() - 1e-5 <= refined["qpos"][model.joint(j).name] <= upper[i].item() + 1e-5
                for i, j in enumerate(model.actuator_trnid[:, 0]))
    target_converged = (reason == "converged" and final_target_error <= config.contact_tolerance
        and audit["max_contact_point_error_m"] <= config.contact_tolerance and final_finger_speed < 0.05)
    checks = {"geometry": bool(audit["passed"]), "joint_limits": bool(legal),
        "trust_region": excursion <= config.max_joint_delta, "drive_projection": drive_projection <= 0.001,
        "mimic_projection": projection_error <= 0.01, "target_convergence": target_converged}
    checks = {name: bool(passed) for name, passed in checks.items()}
    return refined, {"config": asdict(config), "termination": reason, "steps": step + 1,
        "max_joint_excursion_rad": excursion, "mimic_projection_rad": projection_error,
        "drive_limit_projection_rad": drive_projection,
        "final_target_error_m": final_target_error, "final_drive_speed_rad_s": final_speed,
        "final_finger_speed_rad_s": final_finger_speed,
        "exported_target_error_m": audit["max_contact_point_error_m"],
        "target_converged": bool(target_converged), "quality_checks": checks,
        "failure_reasons": [name for name, passed in checks.items() if not passed],
        "raw_joint_pos": raw_joint_pos,
        "simulation_overrides": {"multiccd": False, "contact_condim": 1,
            "equality_solref": [0.002, 1], "joint_limit_solref": [0.002, 1]},
        "geometry": audit, "passed": bool(all(checks.values())),
        "trace_columns": ["time_s", "max_target_error_m", "max_finger_speed_rad_s", "max_joint_delta_rad"], "trace": trace}
