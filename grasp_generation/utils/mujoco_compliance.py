"""MuJoCo binding for independent L20 joint impedance and contact sensing."""

import mujoco
import numpy as np

from .adaptive_compliance import AdaptiveCompliance, ComplianceConfig


FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def independent_limits(robot, name):
    """Intersect a drive's URDF range with the ranges of its mimic children."""
    joint = next(j for j in robot.findall('joint') if j.get('name') == name)
    limit = joint.find('limit')
    lower, upper = float(limit.get('lower')), float(limit.get('upper'))
    for child in robot.findall('joint'):
        mimic = child.find('mimic')
        if mimic is None or mimic.get('joint') != name:
            continue
        ratio, offset = float(mimic.get('multiplier', '1')), float(mimic.get('offset', '0'))
        bounds = child.find('limit')
        if ratio == 0:
            if not float(bounds.get('lower')) <= offset <= float(bounds.get('upper')):
                raise ValueError(f'Infeasible constant mimic: {child.get("name")}')
            continue
        a, b = sorted(((float(bounds.get('lower'))-offset)/ratio,
                       (float(bounds.get('upper'))-offset)/ratio))
        lower, upper = max(lower, a), min(upper, b)
    if lower >= upper:
        raise ValueError(f'Empty feasible drive range: {name}')
    return lower, upper


def independent_velocity(robot, name):
    joints = {j.get('name'): j for j in robot.findall('joint')}
    speed = float(joints[name].find('limit').get('velocity'))
    for joint in joints.values():
        mimic = joint.find('mimic')
        if mimic is not None and mimic.get('joint') == name:
            ratio = abs(float(mimic.get('multiplier', '1')))
            if ratio > 0:
                speed = min(speed, float(joint.find('limit').get('velocity'))/ratio)
    return speed


class MujocoJointCompliance:
    def __init__(self, model, data, robot, reference, mode='adaptive', config=None):
        if mode not in ('fixed', 'adaptive'):
            raise ValueError(f'Unsupported controller: {mode}')
        self.model, self.data, self.mode = model, data, mode
        self.config = config or ComplianceConfig()
        self.object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'object')
        self.joints = np.array(model.actuator_trnid[:, 0], dtype=int)
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        urdf_joints = {j.get('name'): j for j in robot.findall('joint')}
        if any(urdf_joints[n].find('mimic') is not None for n in self.names):
            raise ValueError('Mimic joints cannot be independent drives')
        self.qadr = model.jnt_qposadr[self.joints]
        self.vadr = model.jnt_dofadr[self.joints]
        self.lower = np.array([independent_limits(robot, n)[0] for n in self.names])
        self.upper = np.array([independent_limits(robot, n)[1] for n in self.names])
        model.actuator_ctrlrange[:] = np.column_stack((self.lower, self.upper))
        self.body_finger = np.full(model.nbody, -1, dtype=int)
        for body in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
            for f, finger in enumerate(FINGERS):
                if name.startswith(finger+'_'):
                    self.body_finger[body] = f
        self.finger_index = np.array([FINGERS.index(n.split('_')[0]) for n in self.names])
        self.mass_matrix = np.zeros((model.nv, model.nv))
        self.transmission = np.zeros((model.nv, model.nu))
        self.transmission[self.vadr, np.arange(model.nu)] = 1
        # Reflect passive DIP inertia through its actual mechanical coupling.
        for joint in robot.findall('joint'):
            mimic = joint.find('mimic')
            if mimic is not None:
                col = self.names.index(mimic.get('joint'))
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.get('name'))
                self.transmission[model.jnt_dofadr[jid], col] = float(mimic.get('multiplier', '1'))
        direction = np.array([1. if n.endswith(('_mcp_pitch', '_pip')) or n == 'thumb_mcp' else 0.
                              for n in self.names])
        middle = np.array([n.startswith('middle_') for n in self.names])
        stiffness_multiplier = np.where(middle, self.config.middle_stiffness_multiplier, 1.)
        closure_rate_multiplier = np.where(middle, self.config.middle_closure_rate_multiplier, 1.)
        self.controller = AdaptiveCompliance(
            [reference[n] for n in self.names], self.lower, self.upper,
            np.min(abs(model.actuator_forcerange), axis=1), self.effective_inertia(), direction,
            [independent_velocity(robot, n) for n in self.names], self.config,
            stiffness_multiplier, closure_rate_multiplier,
        )
        # Match the baseline when it lies inside the configured adaptive bounds.
        self.controller.damping[:] = np.clip(.05, self.config.damping_min, self.config.damping_max)
        self.fixed_reference = self.controller.reference.copy()
        self.filtered_force = np.zeros(model.nu)
        self.last = None

    def effective_inertia(self):
        mujoco.mj_fullM(self.model, self.data, self.mass_matrix)
        return np.maximum(np.sum(self.transmission*(self.mass_matrix@self.transmission), axis=0), 1e-9)

    def contacts(self):
        forces = np.zeros(len(FINGERS))
        count, total = 0, 0.
        wrench = np.zeros(6)
        for ci, contact in enumerate(self.data.contact):
            bodies = (self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2])
            if self.object_id not in bodies:
                continue
            mujoco.mj_contactForce(self.model, self.data, ci, wrench)
            normal = max(0., float(wrench[0]))
            if normal <= 1e-5:
                continue
            count, total = count+1, total+normal
            hand_body = bodies[1] if bodies[0] == self.object_id else bodies[0]
            f = self.body_finger[hand_body]
            if f >= 0:
                forces[f] += normal
        return forces, count, total

    def step(self, dt):
        forces, _, _ = self.contacts()
        loads = forces[self.finger_index]
        if self.mode == 'adaptive':
            result = self.controller.update(self.data.qpos[self.qadr], self.data.qvel[self.vadr],
                                             loads, dt, self.effective_inertia())
        else:
            self.filtered_force += (-np.expm1(-dt/self.config.force_filter_time))*(loads-self.filtered_force)
            result = {'reference': self.fixed_reference.copy(), 'stiffness': np.full(self.model.nu, 5.),
                      'damping': np.full(self.model.nu, .05), 'filtered_force': self.filtered_force.copy()}
        self.model.actuator_gainprm[:, 0] = result['stiffness']
        self.model.actuator_biasprm[:, 1] = -result['stiffness']
        self.model.actuator_biasprm[:, 2] = -result['damping']
        self.data.ctrl[:] = result['reference']
        self.last = result
        return result
