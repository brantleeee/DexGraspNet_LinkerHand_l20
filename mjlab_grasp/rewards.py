"""Bounded dense shaping and sustained lift rewards; v1 dynamics/action zero retained."""

import torch
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab_grasp.task import relative_error, rotation_error, stable_success

# All 21 finger joints in linkerhand_l20_right.urdf specify 1 rad/s.
URDF_JOINT_SPEED_LIMIT = 1.0
JOINT_SPEED_LIMIT_RATIO = 0.2
JOINT_SPEED_LIMIT = URDF_JOINT_SPEED_LIMIT * JOINT_SPEED_LIMIT_RATIO


def joint_speed_ok(env):
    """Include passive DIP joints; one fast joint invalidates stable holding."""
    return (
        env.scene["hand"].data.joint_vel.abs().amax(-1) <= JOINT_SPEED_LIMIT
    ).float()


def contact_mask(env):
    return env.scene["grasp_contact"].data.found.flatten(1).any(-1).float()


def retention_gate(env):
    """A smooth gate, so a falling or distant cup cannot earn orientation rewards."""
    distance = relative_error(env).norm(dim=-1)
    return contact_mask(env) * torch.exp(-(distance / 0.03).square())


def capture(env):
    return retention_gate(env)


def orientation(env):
    # Additive shaping avoids the near-zero product reward of experimental v2.
    return retention_gate(env) * torch.exp(-(rotation_error(env) / 0.5).square())


def lift(env):
    obj = env.scene["object"]
    height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    quality = 0.2 + 0.8 * torch.exp(-(rotation_error(env) / 0.5).square())
    return retention_gate(env) * (height / 0.1).clamp(0, 1) * quality


def velocity_error(env):
    obj = env.scene["object"]
    t = env.episode_length_buf * env.step_dt
    u = ((t - 0.5) / 1.0).clamp(0, 1)
    desired = torch.zeros_like(obj.data.root_link_lin_vel_w)
    desired[:, 2] = 0.6 * u * (1 - u)
    return (obj.data.root_link_lin_vel_w - desired).norm(dim=-1)


def stable_hold(env):
    """Paid every stable step: accumulated return rewards long holding, not a peak."""
    obj = env.scene["object"]
    height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    position = torch.exp(-(relative_error(env).norm(dim=-1) / 0.01).square())
    rotation = torch.exp(-(rotation_error(env) / 0.35).square())
    motion = torch.exp(
        -(velocity_error(env) / 0.03).square()
        - (obj.data.root_link_ang_vel_w.norm(dim=-1) / 0.5).square()
    )
    after_lift = (env.episode_length_buf * env.step_dt > 1.5).float()
    return (
        contact_mask(env)
        * after_lift
        * (height / 0.08).clamp(0, 1)
        * position
        * rotation
        * motion
        * joint_speed_ok(env)
    )


def strict_hold(env):
    return contact_mask(env) * stable_success(env) * joint_speed_ok(env)


def slip(env):
    obj = env.scene["object"]
    return torch.tanh(
        (velocity_error(env) / 0.05).square()
        + (obj.data.root_link_ang_vel_w.norm(dim=-1) / 1.0).square()
    )


def position_error(env):
    return torch.tanh((relative_error(env).norm(dim=-1) / 0.02).square())


def orientation_error(env):
    return torch.tanh((rotation_error(env) / 0.5).square())


def raw_saturation(env):
    # Wrapper preserves raw Gaussian actions in v3. Physical targets still clamp.
    excess = (env.action_manager.action.abs() - 1.0).clamp(min=0)
    return (excess / (1.0 + excess)).mean(-1)


def action_change(env):
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.tanh(delta.square().mean(-1))


def joint_motion(env):
    """Penalize the worst speed excess without averaging it across 21 joints."""
    speed = env.scene["hand"].data.joint_vel.abs()
    excess = (speed / JOINT_SPEED_LIMIT - 1.0).clamp(min=0)
    return (excess / (1.0 + excess)).amax(-1)


def failure_event(env):
    # RewardManager multiplies by dt: divide here for a fixed -25 per drop.
    # Timeouts are excluded by termination_manager.terminated.
    return env.termination_manager.terminated.float() / env.step_dt


def settling_progress(env):
    """Dense post-lift signal before the tight stable-hold thresholds are met."""
    obj = env.scene["object"]
    after_lift = (env.episode_length_buf * env.step_dt > 1.5).float()
    position = torch.exp(-(relative_error(env).norm(dim=-1) / 0.03).square())
    orientation_quality = torch.exp(-(rotation_error(env) / 0.7).square())
    linear_quality = torch.exp(-(velocity_error(env) / 0.1).square())
    angular_quality = torch.exp(
        -(obj.data.root_link_ang_vel_w.norm(dim=-1) / 1.0).square()
    )
    return (
        contact_mask(env) * after_lift * position * orientation_quality
        * linear_quality * angular_quality
    )


def target_motion(env):
    """Discourage persistent target chasing after the wrist finishes lifting."""
    term = env.action_manager.get_term("residual")
    after_lift = (env.episode_length_buf * env.step_dt > 1.5).float()
    speed = term._target_step.abs().amax(-1) / env.step_dt
    return after_lift * torch.tanh((speed / JOINT_SPEED_LIMIT).square())


def make_rewards():
    return {
        "capture": RewardTermCfg(func=capture, weight=2.0),
        "orientation": RewardTermCfg(func=orientation, weight=2.0),
        "lift": RewardTermCfg(func=lift, weight=3.0),
        "stable_hold": RewardTermCfg(func=stable_hold, weight=8.0),
        "strict_hold": RewardTermCfg(func=strict_hold, weight=6.0),
        "slip": RewardTermCfg(func=slip, weight=-0.75),
        "position_error": RewardTermCfg(func=position_error, weight=-0.5),
        "orientation_error": RewardTermCfg(func=orientation_error, weight=-0.5),
        "saturation": RewardTermCfg(func=raw_saturation, weight=-0.05),
        "action_change": RewardTermCfg(func=action_change, weight=-0.05),
        "joint_motion": RewardTermCfg(func=joint_motion, weight=-0.5),
        "drop": RewardTermCfg(func=failure_event, weight=-25.0),
    }


def make_rewards_v5():
    """Favor sustained settling over transient capture and lift rewards."""
    rewards = make_rewards()
    for name, weight in {
        "capture": 0.75,
        "orientation": 0.75,
        "lift": 1.5,
        "stable_hold": 10.0,
        "strict_hold": 8.0,
        "slip": -1.0,
        "saturation": -0.25,
        "action_change": -0.2,
        "joint_motion": -1.0,
    }.items():
        rewards[name] = RewardTermCfg(func=rewards[name].func, weight=weight)
    rewards["settling_progress"] = RewardTermCfg(func=settling_progress, weight=4.0)
    rewards["target_motion"] = RewardTermCfg(func=target_motion, weight=-0.25)
    return rewards
