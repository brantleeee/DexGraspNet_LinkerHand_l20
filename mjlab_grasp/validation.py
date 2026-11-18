"""Deterministic, reward-free free-object hold/lift and six-gravity tests."""

from dataclasses import dataclass, asdict

import numpy as np
import torch

from mjlab_grasp.candidates import FINGERS
from mjlab_grasp.metrics import relative_pose, pose_error, advance_hold_steps, required_hold_steps
from mjlab_grasp.physics import Physics, compile_scene, KINEMATIC_ROOT_ARMATURE


GRAVITY = {"gx+": (9.81, 0, 0), "gx-": (-9.81, 0, 0),
           "gy+": (0, 9.81, 0), "gy-": (0, -9.81, 0),
           "gz+": (0, 0, 9.81), "gz-": (0, 0, -9.81), "lift": (0, 0, -9.81)}


@dataclass
class ValidationConfig:
    wrist_motion: str = "kinematic"
    protocol: str = "standard"
    timestep: float = 0.001
    settle_seconds: float = 0.3
    gravity_ramp_seconds: float = 0.2
    prelift_hold_seconds: float = 0.5
    lift_seconds: float = 1.0
    hold_seconds: float = 2.0
    lift_height: float = 0.1
    max_slip: float = 0.005
    max_rotation_deg: float = 15.0
    max_penetration: float = 0.001
    stable_seconds: float = 1.0
    max_linear_speed: float = 0.02
    max_angular_speed: float = 0.3
    min_normal_force: float = 1e-4
    max_joint_limit_error: float = 0.001
    max_mimic_error: float = 0.01
    max_tendon_limit_error: float = 0.0001

    def __post_init__(self):
        if self.wrist_motion not in ("kinematic", "mocap") or self.protocol not in ("standard", "legacy"):
            raise ValueError("Unknown wrist motion or validation protocol")
        for name, value in asdict(self).items():
            if isinstance(value, str):
                continue
            allow_zero = name in ("settle_seconds", "prelift_hold_seconds", "max_penetration", "max_joint_limit_error", "max_mimic_error")
            if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f"Invalid validation parameter: {name}={value}")
        if self.timestep > min(self.gravity_ramp_seconds, self.lift_seconds, self.hold_seconds):
            raise ValueError("Validation phases must each contain at least one simulation step")


def validate(scene, grasps, labels, case, output, backend="mjlab", config=None,
             required_fingers=("thumb", "index", "middle")):
    config = config or ValidationConfig()
    if not grasps or len(labels) != len(grasps) or len(set(labels)) != len(labels):
        raise ValueError("Supply one unique trajectory label per nonempty candidate batch")
    if case not in GRAVITY or not required_fingers or len(set(required_fingers)) != len(required_fingers) or not set(required_fingers) <= set(FINGERS):
        raise ValueError("Invalid case or required finger selection")
    if config.wrist_motion not in ("kinematic", "mocap"):
        raise ValueError(config.wrist_motion)
    scene_name = f"scene_{case}_{backend}.xml"
    model = compile_scene(scene, xml_path=output / scene_name, kinematic_wrist=config.wrist_motion == "kinematic")
    physics = Physics(model, grasps, backend, config.timestep)
    physics.save_scene(output / scene_name)
    oid, wid = model.body("object").id, model.body("wrist").id
    mocap = model.body_mocapid[wid]
    if config.wrist_motion == "kinematic":
        wrist_joint = model.joint("wrist_free").id
        wrist_qadr, wrist_vadr = model.jnt_qposadr[wrist_joint], model.jnt_dofadr[wrist_joint]
    vadr = model.jnt_dofadr[model.joint("object_free").id]
    finger_dofs = model.jnt_dofadr[model.jnt_type == 3]
    n = len(grasps)
    lift_start = config.settle_seconds
    if config.protocol == "standard":
        lift_start += config.gravity_ramp_seconds + config.prelift_hold_seconds
    total = config.settle_seconds + config.hold_seconds
    if config.protocol == "standard":
        total += config.gravity_ramp_seconds
        if case == "lift":
            total += config.prelift_hold_seconds + config.lift_seconds
    lift_end = lift_start + config.lift_seconds if case == "lift" else config.settle_seconds + config.gravity_ramp_seconds
    # Only count complete intervals whose START is after lift/loading ends.
    # Testing t + dt would incorrectly credit the last moving substep.
    stable_start_step = required_hold_steps(lift_end, config.timestep)
    histories = {k: [] for k in ("time", "qpos", "qvel", "ctrl", "qfrc_applied", "mocap_pos", "mocap_quat", "finger_normal_force_N", "loaded_contacts", "slip_m", "rotation_deg", "penetration_m", "stable_hold_s", "joint_limit_error_rad", "mimic_error_rad", "tendon_limit_error_m", "max_finger_speed_rad_s", "wrist_solver_velocity")}
    with physics.stream_context(), torch.no_grad():
        initial_state = {"initial_" + key: physics.read(key).cpu().numpy().copy()
                         for key in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")}
        initial_wrist = physics.read("mocap_pos").clone()
        initial_wrist_pos = physics.read("xpos")[:, wid].clone()
        initial_wrist_quat = physics.read("xquat")[:, wid].clone()
        hand_r = physics.read("xmat")[:, wid].reshape(n, 3, 3)
        obj_r = physics.read("xmat")[:, oid].reshape(n, 3, 3)
        initial_pos, initial_rot = relative_pose(physics.read("xpos")[:, wid], hand_r, physics.read("xpos")[:, oid], obj_r)
        initial_pos, initial_rot = initial_pos.clone(), initial_rot.clone()
        initial_height = physics.read("xpos")[:, oid, 2].clone()
        maxima = torch.zeros((n, 8), device=physics.device)
        depth = physics.contacts()[2]
        maxima[:, 2] = depth
        initial_limit, initial_mimic = physics.joint_errors()
        initial_tendon = physics.tendon_errors()
        maxima[:, 4], maxima[:, 5], maxima[:, 7] = initial_limit, initial_mimic, initial_tendon
        initial_errors = torch.stack((initial_limit, initial_mimic, initial_tendon), -1).cpu().numpy()
        stable_steps = torch.zeros(n, dtype=torch.int64, device=physics.device)
        hold = torch.zeros(n, dtype=torch.float64, device=physics.device)
        longest = hold.clone()
        overflow = torch.zeros(n, dtype=torch.bool, device=physics.device)
        required = [FINGERS.index(f) for f in required_fingers]
        for step in range(round(total / config.timestep)):
            t = step * config.timestep
            ramp = min(1., max(0., (t - config.settle_seconds) / config.gravity_ramp_seconds))
            physics.set_gravity(np.array(GRAVITY[case]) * ramp)
            u = np.clip((t - lift_start) / config.lift_seconds, 0., 1.) if case == "lift" else 0.
            if config.wrist_motion == "kinematic":
                q, v = physics.read("qpos").clone(), physics.read("qvel").clone()
                q[:, wrist_qadr:wrist_qadr+3] = initial_wrist_pos
                q[:, wrist_qadr+2] += config.lift_height * (3*u*u - 2*u*u*u)
                q[:, wrist_qadr+3:wrist_qadr+7] = initial_wrist_quat
                v[:, wrist_vadr:wrist_vadr+6] = 0
                v[:, wrist_vadr+2] = config.lift_height * 6*u*(1-u) / config.lift_seconds
                physics.write("qpos", q)
                physics.write("qvel", v)
                # Prescribe root acceleration too, so finger dynamics include
                # acceleration coupling. Forces act on the ideal arm root only.
                root_force = physics.read("qfrc_applied").clone().zero_()
                if case == "lift" and lift_start <= t < lift_start + config.lift_seconds:
                    acceleration = config.lift_height * 6*(1-2*u) / config.lift_seconds**2
                    root_force[:, wrist_vadr+2] = KINEMATIC_ROOT_ARMATURE * acceleration
                physics.write("qfrc_applied", root_force)
            elif case == "lift":
                wrist = initial_wrist.clone()
                wrist[:, mocap, 2] += config.lift_height * (3 * u*u - 2 * u*u*u)
                physics.write("mocap_pos", wrist)
            physics.step()
            hand_r = physics.read("xmat")[:, wid].reshape(n, 3, 3)
            obj_r = physics.read("xmat")[:, oid].reshape(n, 3, 3)
            pos, rot = relative_pose(physics.read("xpos")[:, wid], hand_r, physics.read("xpos")[:, oid], obj_r)
            slip, angle = pose_error(pos, rot, initial_pos, initial_rot)
            loads, counts, depth, saturated = physics.contacts(config.min_normal_force)
            limit_error, mimic_error = physics.joint_errors()
            tendon_error = physics.tendon_errors()
            finger_speed = physics.read("qvel")[:, finger_dofs].abs().amax(-1)
            overflow |= saturated
            speed = physics.read("qvel")[:, vadr:vadr+3].norm(dim=-1)
            angular = physics.read("qvel")[:, vadr+3:vadr+6].norm(dim=-1)
            height = physics.read("xpos")[:, oid, 2] - initial_height
            stable = ((slip <= config.max_slip) & (angle <= np.radians(config.max_rotation_deg))
                & (speed < config.max_linear_speed) & (angular < config.max_angular_speed)
                & (loads[:, required] > config.min_normal_force).all(-1)
                & (depth <= config.max_penetration) & (step >= stable_start_step)
                & (limit_error <= config.max_joint_limit_error) & (mimic_error <= config.max_mimic_error))
            stable &= tendon_error <= config.max_tendon_limit_error
            if case == "lift":
                stable &= height >= 0.8 * config.lift_height
            stable_steps = advance_hold_steps(stable_steps, stable)
            hold = stable_steps.to(torch.float64) * config.timestep
            longest = torch.maximum(longest, hold)
            maxima[:, 0] = torch.maximum(maxima[:, 0], slip)
            maxima[:, 1] = torch.maximum(maxima[:, 1], angle)
            maxima[:, 2] = torch.maximum(maxima[:, 2], depth)
            maxima[:, 3] = torch.maximum(maxima[:, 3], physics.read("actuator_force").abs().amax(-1))
            maxima[:, 4] = torch.maximum(maxima[:, 4], limit_error)
            maxima[:, 5] = torch.maximum(maxima[:, 5], mimic_error)
            maxima[:, 6] = torch.maximum(maxima[:, 6], finger_speed)
            maxima[:, 7] = torch.maximum(maxima[:, 7], tendon_error)
            if step % max(1, round(0.02/config.timestep)) == 0 or step == round(total/config.timestep)-1:
                values = {"time": np.array(t + config.timestep),
                    "qpos": physics.read("qpos"), "qvel": physics.read("qvel"), "ctrl": physics.read("ctrl"),
                    "qfrc_applied": physics.read("qfrc_applied"),
                    "mocap_pos": physics.read("mocap_pos"), "mocap_quat": physics.read("mocap_quat"),
                    "finger_normal_force_N": loads, "loaded_contacts": counts,
                    "slip_m": slip, "rotation_deg": torch.rad2deg(angle), "penetration_m": depth,
                    "stable_hold_s": hold, "joint_limit_error_rad": limit_error,
                    "mimic_error_rad": mimic_error, "max_finger_speed_rad_s": finger_speed,
                    "tendon_limit_error_m": tendon_error,
                    "wrist_solver_velocity": physics.read("cvel")[:, wid, 3:]}
                for key, value in values.items():
                    histories[key].append(value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else value)
        stats = maxima.cpu().numpy()
        holds, longest_holds = hold.cpu().numpy(), longest.cpu().numpy()
        final_stable_steps = stable_steps.cpu().numpy()
        heights, final_loads = height.cpu().numpy(), loads.cpu().numpy()
        final_counts, overflows = counts.cpu().numpy(), overflow.cpu().numpy()
    results = []
    for i, label in enumerate(labels):
        name = f"{label}_{case}_{backend}"
        trajectory = {key: np.array(value) if key == "time" else np.array(value)[:, i] for key, value in histories.items()}
        trajectory.update({key: value[i] for key, value in initial_state.items()})
        np.savez_compressed(output / f"{name}.npz", **trajectory)
        legacy_pass = bool(stats[i, 0] <= config.max_slip and stats[i, 1] <= np.radians(config.max_rotation_deg) and final_counts[i].sum() > 0)
        minimum_steps = required_hold_steps(config.stable_seconds, config.timestep)
        passed = bool(legacy_pass and final_stable_steps[i] >= minimum_steps
                      and stats[i, 2] <= config.max_penetration and not overflows[i]
                      and stats[i, 4] <= config.max_joint_limit_error and stats[i, 5] <= config.max_mimic_error
                      and stats[i, 7] <= config.max_tendon_limit_error)
        results.append({"variant": label, "backend": backend, "case": case, "controller": "fixed_pd",
            "wrist_motion": config.wrist_motion,
            "wrist_motion_model": "prescribed freejoint pose/velocity, root armature 1e6; ideal arm" if config.wrist_motion == "kinematic" else "historical mocap pose updates; tangential velocity missing",
            "config": asdict(config), "max_relative_slip_mm": float(stats[i, 0] * 1000),
            "max_rotation_deg": float(np.degrees(stats[i, 1])), "peak_object_contact_depth_mm": float(stats[i, 2] * 1000),
            "max_actuator_torque_Nm": float(stats[i, 3]), "final_height_gain_m": float(heights[i]),
            "max_joint_limit_error_rad": float(stats[i, 4]), "max_mimic_error_rad": float(stats[i, 5]),
            "max_finger_speed_rad_s": float(stats[i, 6]),
            "max_tendon_limit_error_m": float(stats[i, 7]),
            "initial_joint_limit_error_rad": float(initial_errors[i, 0]),
            "initial_mimic_error_rad": float(initial_errors[i, 1]),
            "initial_tendon_limit_error_m": float(initial_errors[i, 2]),
            "final_finger_normal_force_N": dict(zip(FINGERS, map(float, final_loads[i]))),
            "final_loaded_contacts": int(final_counts[i].sum()),
            "final_stable_hold_s": float(holds[i]), "longest_stable_hold_s": float(longest_holds[i]),
            "final_stable_steps": int(final_stable_steps[i]), "required_stable_steps": minimum_steps,
            "stable_window_start_step": stable_start_step,
            "contact_sensor_saturated": bool(overflows[i]), "legacy_pose_pass": legacy_pass,
            "dynamics_pass": passed, "trajectory": f"{name}.npz", "scene": scene_name,
            "object_support": "none", "required_fingers": list(required_fingers)})
    return results
