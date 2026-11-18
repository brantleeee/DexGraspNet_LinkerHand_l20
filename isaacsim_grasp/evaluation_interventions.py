"""Explicit evaluation-only changes; do not mutate trained env/controller code."""
import math
import numpy as np
from .diagnostics import PhysicsDiagnostics


def validate_preload_gain(gain):
    if not math.isfinite(gain) or not 0<=gain<=1:
        raise ValueError('preload-gain must be finite and in [0, 1]')


def scale_preload_action(action,gain,step,action_start,gravity_start):
    """Scale tanh-space requests only; leave target rate limiting to the env.

    Returns an equivalent raw command and the multiplier applied this tick.
    Multiplier one returns raw actions exactly, including saturated inputs.
    """
    validate_preload_gain(gain)
    raw=np.asarray(action,dtype=np.float32)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite policy action')
    active=action_start<=step<gravity_start
    applied=gain if active else 1.
    if applied==1:return raw.copy(),applied
    return np.arctanh(applied*np.tanh(raw.astype(np.float64))).astype(np.float32),applied


class InterventionDiagnostics(PhysicsDiagnostics):
    """Keep both actor outputs and commands sent to the unchanged environment."""
    def __init__(self,env):
        self.policy_raw_action=np.zeros((env.n,len(env.controller.independent)),np.float32)
        self.request_gain=1.
        super().__init__(env)

    def begin(self):
        self.policy_raw_action[:]=0;self.request_gain=1.
        super().begin()

    def set_policy_action(self,action,gain):
        self.policy_raw_action=np.asarray(action,dtype=np.float32).copy()
        self.request_gain=float(gain)

    def capture(self,initial=False):
        super().capture(initial)
        self.frames[-1].update(policy_raw_action=self.policy_raw_action.copy(),
            policy_requested_residual=self.env.cfg['residual_scale']*np.tanh(self.policy_raw_action),
            preload_request_gain=np.full(self.env.n,self.request_gain,np.float32))
