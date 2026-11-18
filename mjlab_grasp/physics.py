"""Same compiled model and control law on mjlab Simulation or C MuJoCo.

The CPU backend is a comparison oracle, never a fallback for a failed GPU run.
"""

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from mjlab_grasp.candidates import FINGERS, initialize_data, wrist_pose


CONTACT_SLOTS = 32
KINEMATIC_ROOT_ARMATURE = 1e6


def compile_scene(scene, fixed_object=False, torque_control=False, margin=0.0, torque_damping=0.03, xml_path=None, kinematic_wrist=False):
    scene = copy.deepcopy(scene)
    if kinematic_wrist:
        if fixed_object:
            raise ValueError("Refinement uses a fixed wrist")
        wrist = scene.find("worldbody/body[@name='wrist']")
        wrist.attrib.pop("mocap", None)
        # Prescribed arm motion with negligible within-substep root response.
        # Unlike mocap teleportation, freejoint qvel enters contact Jacobians.
        if wrist.find("inertial") is None:
            ET.SubElement(wrist, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
        root_joint = wrist.find("joint[@name='wrist_free']")
        if root_joint is None:
            if wrist.find("joint") is not None or wrist.find("freejoint") is not None:
                raise ValueError("Unsupported wrist root joint; expected mocap or wrist_free")
            root_joint = ET.SubElement(wrist, "joint", name="wrist_free", type="free")
        if root_joint.get("type") != "free":
            raise ValueError("wrist_free must be a free joint")
        root_joint.set("armature", str(KINEMATIC_ROOT_ARMATURE))
        root_joint.set("damping", "0")
        root_joint.set("limited", "false")
    if fixed_object:
        obj = scene.find("worldbody/body[@name='object']")
        free_joint = obj.find("freejoint")
        if free_joint is None:
            # MjSpec.to_xml serializes <freejoint> as <joint type="free">.
            free_joint = obj.find("joint[@type='free']")
        if free_joint is None:
            raise ValueError("Refinement requires a source scene with a free object joint")
        obj.remove(free_joint)
        # MuJoCo-Warp mesh MULTICCD currently rejects nonzero contact margins.
        # Only the refinement model uses this flag; free-object validation keeps
        # the original collision algorithm, on both CPU and GPU.
        option = scene.find("option")
        if option is None:
            option = ET.SubElement(scene, "option")
        flag = option.find("flag")
        if flag is None:
            flag = ET.SubElement(option, "flag")
        flag.set("multiccd", "disable")
        # URDF mimic is a mechanical coupling. The default soft 20 ms
        # equality is too compliant under virtual point forces on tiny links.
        for equality in scene.findall("equality/joint"):
            equality.set("solref", "0.002 1")
            equality.set("solimp", "0.99 0.99 0.001")
        for joint in scene.findall("worldbody/.//joint"):
            joint.set("solreflimit", "0.002 1")
    if torque_control:
        for actuator in scene.find("actuator"):
            effort = actuator.get("forcerange")
            actuator.set("gainprm", "1")
            # Keep damping in the implicit integrator, rather than explicitly
            # feeding back torque on this very low-inertia hand.
            actuator.set("biasprm", f"0 0 {-torque_damping:g}")
            actuator.set("ctrlrange", effort)
    for geom in scene.findall(".//geom"):
        if geom.get("contype", "1") != "0":
            geom.set("margin", str(margin))
            if fixed_object:
                geom.set("friction", "0 0 0")
                geom.set("condim", "1")
    # Locate the root of each complete finger subtree, including thumb CMC bodies.
    roots = {}
    shadow_roots = dict(zip(FINGERS, ("robot0:thbase", "robot0:ffknuckle", "robot0:mfknuckle", "robot0:rfknuckle", "robot0:lfmetacarpal")))
    body_names = {body.get("name") for body in scene.findall(".//body")}
    if "robot0:palm" in body_names:
        if not set(shadow_roots.values()) <= body_names:
            raise ValueError("Incomplete ShadowHand finger structure")
        roots.update(shadow_roots)
    for body in scene.findall(".//body"):
        name = body.get("name", "")
        for finger in FINGERS:
            if name.startswith(finger + "_") and finger not in roots:
                roots[finger] = name
    # Recompiling an exported scene must not duplicate our native sensors.
    reserved = {f"{finger}_{field}" for finger in roots for field in ("found", "force")} | {"object_min_dist"}
    sensors = scene.find("sensor")
    if sensors is not None:
        for sensor in list(sensors):
            if sensor.get("name") in reserved:
                sensors.remove(sensor)
    spec = mujoco.MjSpec.from_string(ET.tostring(scene, encoding="unicode"))
    for finger, root in roots.items():
        for field, bit in [("found", 1), ("force", 2)]:
            spec.add_sensor(
                name=f"{finger}_{field}", type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_XBODY, objname=root,
                reftype=mujoco.mjtObj.mjOBJ_BODY, refname="object",
                intprm=[bit, 0, CONTACT_SLOTS],
            )
    spec.add_sensor(
        name="object_min_dist", type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY, objname="wrist",
        reftype=mujoco.mjtObj.mjOBJ_BODY, refname="object", intprm=[8, 1, 1],
    )
    model = spec.compile()
    if xml_path is not None:
        xml_path.write_text(spec.to_xml())
    return model


class Physics:
    def __init__(self, model, grasps, backend="mjlab", timestep=0.001, gravity=(0, 0, 0)):
        self.model = model
        self.backend = backend
        self.dt = timestep
        model.opt.timestep = timestep
        model.opt.gravity[:] = gravity
        model.opt.iterations = 100
        model.opt.ls_iterations = 50
        model.opt.tolerance = 1e-8
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # Match every option overridden by the GPU SimulationCfg on CPU too.
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
        model.opt.impratio = 1.0
        model.opt.ls_tolerance = 0.01
        model.opt.ccd_iterations = 50
        self.n = len(grasps)
        qpos, ctrl = initialize_data(model, grasps)
        default = mujoco.MjData(model)
        mocap_pos = np.repeat(default.mocap_pos[None], self.n, axis=0)
        mocap_quat = np.repeat(default.mocap_quat[None], self.n, axis=0)
        wrist_mocap = model.body_mocapid[model.body("wrist").id]
        wrist_free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_free")
        for i, grasp in enumerate(grasps):
            if wrist_free >= 0:
                qa = model.jnt_qposadr[wrist_free]
                if "WRJTx" in grasp["qpos"]:
                    position, quat = wrist_pose(grasp)
                    qpos[i, qa:qa+3], qpos[i, qa+3:qa+7] = position, quat
            elif "WRJTx" in grasp["qpos"]:
                mocap_pos[i, wrist_mocap], mocap_quat[i, wrist_mocap] = wrist_pose(grasp)
        self.device = "cuda:0" if backend == "mjlab" else "cpu"
        self.sim = None
        if backend == "mjlab":
            from mjlab.sim import Simulation, SimulationCfg, MujocoCfg

            if not torch.cuda.is_available():
                raise RuntimeError("mjlab requires CUDA; CPU comparison must be requested explicitly")
            self.sim = Simulation(
                self.n, SimulationCfg(nconmax=512, njmax=2048, contact_sensor_maxmatch=2048,
                    mujoco=MujocoCfg(timestep=timestep, gravity=gravity, iterations=100,
                                     tolerance=1e-8, ls_iterations=50)), model=model,
                device=self.device,
            )
            # Use the Warp stream for all torch kernels touching zero-copy state.
            import warp as wp
            self.stream = torch.cuda.ExternalStream(wp.get_stream(self.sim.wp_device).cuda_stream)
            self.data = self.sim.data
            self.sim.reset()
            with torch.cuda.stream(self.stream):
                self.data.qpos[:] = self.tensor(qpos)
                self.data.qvel.zero_()
                self.data.ctrl[:] = self.tensor(ctrl)
                self.data.mocap_pos[:] = self.tensor(mocap_pos)
                self.data.mocap_quat[:] = self.tensor(mocap_quat)
        elif backend == "cpu":
            self.cpu = [mujoco.MjData(model) for _ in grasps]
            for i, data in enumerate(self.cpu):
                data.qpos[:] = qpos[i]
                data.ctrl[:] = ctrl[i]
                data.mocap_pos[:] = mocap_pos[i]
                data.mocap_quat[:] = mocap_quat[i]
        else:
            raise ValueError(backend)
        self.forward()

    def tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def save_scene(self, path):
        """Persist runtime options, including dt, rather than template defaults."""
        spec = mujoco.MjSpec.from_file(str(path))
        for name in dir(self.model.opt):
            if name.startswith("_") or not hasattr(spec.option, name):
                continue
            value = getattr(self.model.opt, name)
            if isinstance(value, np.ndarray):
                getattr(spec.option, name)[:] = value
            elif np.isscalar(value):
                setattr(spec.option, name, value)
        path.write_text(spec.to_xml())

    def read(self, field):
        if self.sim is not None:
            return getattr(self.data, field)[:]
        return self.tensor(np.array([getattr(d, field) for d in self.cpu]))

    def write(self, field, value):
        if self.sim is not None:
            getattr(self.data, field)[:] = value
        else:
            array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for i, data in enumerate(self.cpu):
                getattr(data, field)[:] = array[i]

    def step(self):
        if self.sim is not None:
            self.sim.step()
        else:
            for data in self.cpu:
                mujoco.mj_step(self.model, data)
        self.forward()
        if not torch.isfinite(self.read("qpos")).all() or not torch.isfinite(self.read("qvel")).all():
            raise RuntimeError("Nonfinite simulation state")
        if self.sim is None and any(w.number for d in self.cpu for w in d.warning):
            raise RuntimeError("MuJoCo simulation warning")
        if self.sim is not None and self.read("overflow").any():
            raise RuntimeError("MuJoCo-Warp capacity overflow; increase simulation/contact capacity")

    def set_gravity(self, gravity):
        self.model.opt.gravity[:] = gravity
        if self.sim is not None:
            import warp as wp
            wp.to_torch(self.sim.wp_model.opt.gravity)[:] = self.tensor(gravity)

    def forward(self):
        if self.sim is not None:
            self.sim.forward()
        else:
            for data in self.cpu:
                mujoco.mj_forward(self.model, data)

    def stream_context(self):
        from contextlib import nullcontext
        return torch.cuda.stream(self.stream) if self.sim is not None else nullcontext()

    def sensor(self, name):
        s = self.model.sensor(name)
        return self.read("sensordata")[:, s.adr[0]:s.adr[0] + s.dim[0]]

    def contacts(self, threshold=1e-4):
        loads, counts, saturated = [], [], []
        for finger in FINGERS:
            found = self.sensor(f"{finger}_found") > 0
            force = self.sensor(f"{finger}_force").reshape(self.n, CONTACT_SLOTS, 3)[:, :, 0].clamp(min=0)
            loads.append((force * found).sum(-1))
            counts.append(((force > threshold) & found).sum(-1))
            saturated.append(found.sum(-1) >= CONTACT_SLOTS)
        depth = (-self.sensor("object_min_dist")[:, 0]).clamp(min=0)
        return torch.stack(loads, -1), torch.stack(counts, -1), depth, torch.stack(saturated, -1).any(-1)

    def joint_errors(self):
        model = self.model
        q = self.read("qpos")
        ids = np.flatnonzero((model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE) & model.jnt_limited)
        positions = q[:, model.jnt_qposadr[ids]]
        bounds = self.tensor(model.jnt_range[ids])
        limit_error = torch.maximum(bounds[:, 0] - positions, positions - bounds[:, 1]).clamp(min=0).amax(-1)
        mimic_error = torch.zeros(self.n, device=self.device)
        for e in range(model.neq):
            if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
                child, parent = model.eq_obj1id[e], model.eq_obj2id[e]
                x = q[:, model.jnt_qposadr[parent]]
                expected = sum(float(c) * x**i for i, c in enumerate(model.eq_data[e, :5]))
                mimic_error = torch.maximum(mimic_error, (q[:, model.jnt_qposadr[child]] - expected).abs())
        return limit_error, mimic_error

    def tendon_errors(self):
        """Limited tendon lengths use metres, separately from joint radians."""
        ids = np.flatnonzero(self.model.tendon_limited)
        if not len(ids):
            return torch.zeros(self.n, device=self.device)
        lengths = self.read("ten_length")[:, ids]
        bounds = self.tensor(self.model.tendon_range[ids])
        return torch.maximum(bounds[:, 0] - lengths, lengths - bounds[:, 1]).clamp(min=0).amax(-1)


def independent_transmission(model):
    """dq = T du for linear L20 equalities; include passive DIP work."""
    joint_ids = model.actuator_trnid[:, 0].astype(int)
    vadr = model.jnt_dofadr[joint_ids]
    matrix = np.zeros((model.nv, model.nu))
    matrix[vadr, np.arange(model.nu)] = 1
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        child, parent = model.eq_obj1id[e], model.eq_obj2id[e]
        if parent not in joint_ids or np.any(model.eq_data[e, 2:5] != 0):
            raise ValueError("Only linear driven-parent joint equalities are supported")
        matrix[model.jnt_dofadr[child], np.flatnonzero(joint_ids == parent)[0]] = model.eq_data[e, 1]
    return matrix, model.jnt_qposadr[joint_ids], vadr


def point_jacobian(physics, body_ids, local_points):
    """Batched world point Jacobian for this fixed-wrist revolute hand.

    Object freejoint columns remain zero; no CPU Jacobian calls in the GPU loop.
    """
    model = physics.model
    ids = np.asarray(body_ids)
    xpos = physics.read("xpos")[:, ids]
    rotation = physics.read("xmat")[:, ids].reshape(physics.n, len(ids), 3, 3)
    points = xpos + torch.einsum("bcij,cj->bci", rotation, physics.tensor(local_points))
    jac = torch.zeros((physics.n, len(ids), 3, model.nv), device=physics.device)
    anchors, axes = physics.read("xanchor"), physics.read("xaxis")
    for c, body_id in enumerate(ids):
        body = int(body_id)
        while body:
            for jid in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
                if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("Refinement hand must have a fixed wrist and hinge joints")
                jac[:, c, :, model.jnt_dofadr[jid]] = torch.linalg.cross(
                    axes[:, jid], points[:, c] - anchors[:, jid], dim=-1)
            body = int(model.body_parentid[body])
    return points, jac
