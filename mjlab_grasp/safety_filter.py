"""Optional post-lift command filter for an existing v4 policy."""

import torch


class HoldTargetFilter:
    def __init__(self, env, action_term, rate_rad_s=0.2):
        self.env = env
        self.term = action_term
        self.rate_rad_s = rate_rad_s
        self.target = action_term._offset.clone()
        self.last_episode_length = env.episode_length_buf.clone()
        self.last_step = torch.zeros_like(self.target)

    @torch.inference_mode()
    def __call__(self, raw_actions):
        elapsed = self.env.episode_length_buf * self.env.step_dt
        reset = self.env.episode_length_buf < self.last_episode_length
        self.target[reset] = self.term._offset[reset]
        self.last_episode_length = self.env.episode_length_buf.clone()
        desired = self.term._offset + self.term._scale * raw_actions.clamp(-1, 1)
        desired = desired.clamp(self.term._clip[:, :, 0], self.term._clip[:, :, 1])
        hold = elapsed >= 1.5
        delta = (desired - self.target).clamp(
            -self.rate_rad_s * self.env.step_dt,
            self.rate_rad_s * self.env.step_dt,
        )
        filtered = torch.where(hold[:, None], self.target + delta, desired)
        self.last_step = filtered - self.target
        self.target = filtered
        return (filtered - self.term._offset) / self.term._scale

    @torch.inference_mode()
    def reset(self, env_ids):
        self.target[env_ids] = self.term._offset[env_ids]
        self.last_step[env_ids] = 0
        self.last_episode_length[env_ids] = 0
