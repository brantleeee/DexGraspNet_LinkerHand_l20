"""Static grasp action-zero task: retain a free cup while lifting the wrist."""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mujoco
import torch
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data/mjlab_grasp/scale_0p06_candidate_7"


class StaticResidualAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions.clamp(-1, 1))


@dataclass(kw_only=True)
class StaticResidualActionCfg(JointPositionActionCfg):
    def build(self, env):
        return StaticResidualAction(self, env)


class SmoothResidualAction(JointPositionAction):
    """Bound residuals smoothly and limit target speed to 1 rad/s."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._filtered_target = self._offset.clone()
        self._processed_actions = self._offset.clone()
        self._max_delta = env.step_dt * 1.0

    def process_actions(self, actions):
        super().process_actions(torch.tanh(actions))
        self._filtered_target += (
            self._processed_actions - self._filtered_target
        ).clamp(-self._max_delta, self._max_delta)
        self._processed_actions = self._filtered_target.clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._filtered_target[ids] = self._offset[ids]
        self._processed_actions[ids] = self._offset[ids]


@dataclass(kw_only=True)
class SmoothResidualActionCfg(JointPositionActionCfg):
    def build(self, env):
        return SmoothResidualAction(self, env)


class RateLimitedResidualAction(StaticResidualAction):
    """Smooth policy commands and cap the position target's change per step."""

    capture_rate_rad_s = 0.8
    hold_rate_rad_s = 0.2
    filter_time_s = 0.08

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._filtered_target = self._offset.clone()
        self._processed_actions = self._offset.clone()
        self._target_step = torch.zeros_like(self._filtered_target)

    def process_actions(self, actions):
        super().process_actions(actions)
        requested = self._processed_actions
        dt = self._env.step_dt
        alpha = dt / (self.filter_time_s + dt)
        smoothed = self._filtered_target + alpha * (requested - self._filtered_target)
        elapsed = self._env.episode_length_buf * dt
        rate = torch.where(
            elapsed < 1.5, self.capture_rate_rad_s, self.hold_rate_rad_s
        )[:, None]
        limit = rate * dt
        delta = (smoothed - self._filtered_target).clamp(min=-limit, max=limit)
        self._target_step = delta
        self._filtered_target = self._filtered_target + delta
        self._processed_actions = self._filtered_target

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._filtered_target[ids] = self._offset[ids]
        self._processed_actions[ids] = self._offset[ids]
        self._target_step[ids] = 0


@dataclass(kw_only=True)
class RateLimitedResidualActionCfg(JointPositionActionCfg):
    def build(self, env):
        return RateLimitedResidualAction(self, env)


def target_state(env):
    term = env.action_manager.get_term("residual")
    return ((term._processed_actions - term._offset) / term._scale).clamp(-1, 1)


def target_residual(env):
    term = env.action_manager.get_term("residual")
    return (term._processed_actions - term._offset) / 0.2


def rotation_error(env):
    obj = env.scene["object"]
    dot = (obj.data.root_link_quat_w * obj.data.default_root_state[:, 3:7]).sum(-1)
    return 2 * torch.acos(dot.abs().clamp(0, 1))


def stable_pose_reward(env):
    position = torch.sum(relative_error(env) ** 2, dim=-1) / 0.01**2
    rotation = (rotation_error(env) / 0.35) ** 2
    return torch.exp(-position - rotation)


def stable_lift_reward(env):
    obj = env.scene["object"]
    height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height / 0.1).clamp(0, 1) * stable_pose_reward(env)


def motion_penalty(env):
    obj = env.scene["object"]
    t = env.episode_length_buf * env.step_dt
    u = ((t - 0.5) / 1.0).clamp(0, 1)
    expected_v = torch.zeros_like(obj.data.root_link_lin_vel_w)
    expected_v[:, 2] = 0.6 * u * (1 - u)
    linear = ((obj.data.root_link_lin_vel_w - expected_v) ** 2).sum(-1)
    angular = (obj.data.root_link_ang_vel_w**2).sum(-1)
    return (linear / 0.05**2 + angular / 1.0**2).clamp(max=20)


def stable_success(env):
    obj = env.scene["object"]
    return (
        strict_success(env)
        * (
            (obj.data.root_link_lin_vel_w.norm(dim=-1) < 0.02)
            & (obj.data.root_link_ang_vel_w.norm(dim=-1) < 0.3)
            & (env.episode_length_buf * env.step_dt > 1.5)
        ).float()
    )


def residual_effort(env):
    # Penalize the actual unbounded Gaussian command, before tanh saturation.
    return env.action_manager.action.square().mean(-1)


def load_spec(path, actuator_damping=None):
    spec = mujoco.MjSpec.from_file(str(path))
    if actuator_damping is not None:
        for actuator in spec.actuators:
            actuator.biasprm[2] = -actuator_damping
    return spec


def relative_error(env):
    hand, obj = env.scene["hand"], env.scene["object"]
    initial_delta = (
        obj.data.default_root_state[:, :3] - hand.data.default_root_state[:, :3]
    )
    return obj.data.root_link_pos_w - hand.data.root_link_pos_w - initial_delta


def phase(env):
    t = env.episode_length_buf * env.step_dt
    return torch.stack((t / 2.5, ((t - 0.5) / 1.0).clamp(0, 1)), dim=-1)


def object_state(env):
    obj = env.scene["object"]
    return torch.cat(
        (
            relative_error(env),
            obj.data.root_link_quat_w,
            obj.data.root_link_lin_vel_w,
            obj.data.root_link_ang_vel_w,
        ),
        dim=-1,
    )


def contact_state(env):
    data = env.scene["grasp_contact"].data
    return torch.cat(
        (data.found.float().flatten(1), data.force.flatten(1).clamp(-30, 30) / 10),
        dim=-1,
    )


def pose_hold_reward(env):
    return torch.exp(-torch.sum(relative_error(env) ** 2, dim=-1) / 0.02**2)


def lift_reward(env):
    obj = env.scene["object"]
    height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height / 0.1).clamp(0, 1) * pose_hold_reward(env)


def posture_penalty(env):
    h = env.scene["hand"]
    return ((h.data.joint_pos - h.data.default_joint_pos) ** 2).mean(-1)


def strict_success(env):
    obj = env.scene["object"]
    height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    rotation = 2 * torch.acos(obj.data.root_link_quat_w[:, 0].abs().clamp(0, 1))
    return (
        (height > 0.08)
        & (relative_error(env).norm(dim=-1) < 0.005)
        & (rotation < 0.2618)
        & (env.episode_length_buf * env.step_dt > 1.5)
    ).float()


def dropped(env):
    obj = env.scene["object"]
    height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (relative_error(env).norm(dim=-1) > 0.1) | (height < -0.08)


def lift_wrist(env, env_ids):
    if env_ids is None:
        env_ids = slice(None)
    hand = env.scene["hand"]
    target = hand.data.default_root_state[env_ids, :7].clone()
    target[:, :3] += env.scene.env_origins[env_ids]
    t = env.episode_length_buf[env_ids] * env.step_dt
    s = ((t - 0.5) / 1.0).clamp(0, 1)
    target[:, 2] += 0.1 * (3 * s * s - 2 * s * s * s)
    hand.write_mocap_pose_to_sim(target, env_ids=env_ids)


def free_thumb_parameters(ref, hand_xml):
    """Use feasible joint-range midpoints, independent of the static thumb angles."""
    initial = ref["joint_pos"].copy()
    offsets, scales = {}, {}
    for name in ref["drive_names"]:
        if name.startswith("thumb_"):
            low, high = ref["drive_limits"][name]
            offsets[name] = initial[name] = (low + high) / 2
            scales[name] = (high - low) / 2
        else:
            offsets[name] = initial[name]
            scales[name] = 0.2
    # Match the actual exported mimic equality, including its offset/multiplier.
    for equality in ET.parse(hand_xml).getroot().findall("equality/joint"):
        child, parent = equality.get("joint1"), equality.get("joint2")
        if child.startswith("thumb_") and parent:
            coefficients = [
                float(x) for x in equality.get("polycoef", "0 1 0 0 0").split()
            ]
            initial[child] = sum(
                c * initial[parent] ** i for i, c in enumerate(coefficients)
            )
    return initial, offsets, scales


def make_env(reference_dir=DEFAULT_REFERENCE, num_envs=64, version="v1", actuator_damping=None):
    if version not in ("v1", "v2", "v3", "v4", "v5"):
        raise ValueError(version)
    directory = Path(reference_dir)
    ref = json.loads((directory / "reference.json").read_text())
    if ref.get("collision_representation") == "mesh_sdf":
        from mjlab_grasp.sdf import digest
        for name, expected in ref["asset_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"SDF reference XML changed: {name}")
        for path, expected in ref["sdf_mesh_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"SDF source mesh changed: {path}")
    if not 0 < ref["scale"] < float("inf") or not isinstance(ref["source_index"], int):
        raise ValueError("This task requires a positive finite scale and an indexed reference")
    initial = ref["joint_pos"].copy()
    offsets, scales = 0.0, 0.2
    if version in ("v4", "v5"):
        initial, offsets, scales = free_thumb_parameters(ref, directory / "hand.xml")
    hand = EntityCfg(
        spec_fn=partial(load_spec, directory / "hand.xml", actuator_damping),
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(ref["wrist_pos"]),
            rot=tuple(ref["wrist_quat"]),
            joint_pos=initial,
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=tuple(ref["drive_names"]),
                    command_field="position",
                ),
            )
        ),
    )
    obj = EntityCfg(
        spec_fn=partial(load_spec, directory / "object.xml"),
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(ref.get("object_pose", [0, 0, 0, 1, 0, 0, 0])[:3]),
            rot=tuple(ref.get("object_pose", [0, 0, 0, 1, 0, 0, 0])[3:]),
        ),
    )
    terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("hand")}
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("hand")}
        ),
        "object": ObservationTermCfg(func=object_state),
        "contacts": ObservationTermCfg(func=contact_state),
        "phase": ObservationTermCfg(func=phase),
        "last_action": ObservationTermCfg(func=mdp.last_action),
    }
    if version in ("v3", "v4", "v5"):
        terms["last_action"] = ObservationTermCfg(
            func=lambda env: env.action_manager.action.clamp(-1, 1)
        )
    if version == "v2":
        terms["target_residual"] = ObservationTermCfg(func=target_residual)
        terms["phase"] = ObservationTermCfg(
            func=lambda env: phase(env) * torch.tensor([0.5, 1.0], device=env.device)
        )
    if version == "v5":
        terms["target_state"] = ObservationTermCfg(func=target_state)
    sensor = ContactSensorCfg(
        name="grasp_contact",
        primary=ContactMatch(mode="subtree", pattern="wrist", entity="hand"),
        secondary=ContactMatch(mode="body", pattern="object", entity="object"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    resets = {}
    for entity in ["hand", "object"]:
        resets["reset_" + entity] = EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(entity),
            },
        )
    resets["reset_joints"] = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("hand"),
        },
    )
    resets["lift"] = EventTermCfg(func=lift_wrist, mode="step")
    from mjlab_grasp.rewards import make_rewards, make_rewards_v5

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=0.8,
            entities={"hand": hand, "object": obj},
            sensors=(sensor,),
        ),
        actions={
            "residual": (
                SmoothResidualActionCfg if version == "v2" else
                RateLimitedResidualActionCfg if version == "v5" else StaticResidualActionCfg
            )(
                entity_name="hand",
                actuator_names=tuple(ref["drive_names"]),
                use_default_offset=version not in ("v4", "v5"),
                offset=offsets,
                scale=scales,
                clip={n: tuple(v) for n, v in ref["drive_limits"].items()},
            )
        },
        observations={
            name: ObservationGroupCfg(terms=terms.copy(), enable_corruption=False)
            for name in ["actor", "critic"]
        },
        rewards=(
            (make_rewards_v5() if version == "v5" else make_rewards())
            if version in ("v3", "v4", "v5")
            else (
                {
                    "hold": RewardTermCfg(func=pose_hold_reward, weight=2.0),
                    "lift": RewardTermCfg(func=lift_reward, weight=5.0),
                    "success": RewardTermCfg(func=strict_success, weight=3.0),
                    "posture": RewardTermCfg(func=posture_penalty, weight=-0.1),
                    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
                }
                if version == "v1"
                else {
                    "hold": RewardTermCfg(func=stable_pose_reward, weight=2.0),
                    "lift": RewardTermCfg(func=stable_lift_reward, weight=5.0),
                    "success": RewardTermCfg(func=stable_success, weight=5.0),
                    "motion": RewardTermCfg(func=motion_penalty, weight=-0.1),
                    "command": RewardTermCfg(func=residual_effort, weight=-0.05),
                    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
                    "posture": RewardTermCfg(func=posture_penalty, weight=-0.1),
                }
            )
        ),
        events=resets,
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "dropped": TerminationTermCfg(func=dropped),
        },
        decimation=10,
        episode_length_s=5.0 if version in ("v2", "v3", "v4", "v5") else 2.5,
        sim=SimulationCfg(
            nconmax=256,
            njmax=1024,
            # 22 hand hulls x 27 cup hulls can exceed the default 64 matches.
            contact_sensor_maxmatch=1024,
            mujoco=MujocoCfg(timestep=0.002, iterations=30, ls_iterations=20),
        ),
        viewer=ViewerConfig(
            entity_name="hand",
            body_name="wrist",
            distance=0.5,
            elevation=-20,
            azimuth=-45,
        ),
    )


def make_agent(version="v1"):
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.3,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(128, 128), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=3e-4,
            entropy_coef=0.005,
        ),
        num_steps_per_env=32,
        max_iterations=2000,
        save_interval=100,
        experiment_name="l20_static_residual_006",
        logger="tensorboard",
        upload_model=False,
        clip_actions=None if version in ("v2", "v3", "v4", "v5") else 1.0,
    )
