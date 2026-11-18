"""Object retention metrics independent of policy rewards and world origin."""

import torch
import math


def relative_pose(hand_pos, hand_rotation, object_pos, object_rotation):
    inverse = hand_rotation.transpose(-1, -2)
    return (inverse @ (object_pos - hand_pos).unsqueeze(-1)).squeeze(-1), inverse @ object_rotation


def pose_error(position, rotation, initial_position, initial_rotation):
    slip = (position - initial_position).norm(dim=-1)
    delta = initial_rotation.transpose(-1, -2) @ rotation
    cosine = ((delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return slip, torch.acos(cosine)


def advance_hold(previous, stable, dt):
    return torch.where(stable, previous + dt, torch.zeros_like(previous))


def advance_hold_steps(previous, stable):
    return torch.where(stable, previous + 1, torch.zeros_like(previous))


def required_hold_steps(seconds, dt):
    # Count discrete samples; avoid accumulated float32 time at the boundary.
    return math.ceil(seconds / dt - 1e-9)
