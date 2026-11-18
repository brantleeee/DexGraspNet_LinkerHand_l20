"""Bounded, contact-adaptive joint impedance, independent of the simulator.

The caller supplies nonnegative contact loads (N), joint state and diagonal
effective inertia. For the L20 adapter a finger's measured normal load is shared
by its independent joints. Mimic joints must never be included as drives.

tau = K(t) * (q_reference(t) - q) - D(t) * q_velocity

Gains and equilibrium offsets are rate/position bounded; loads are low-pass filtered.
This is a variable-impedance heuristic, not a passivity or stability proof.
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class ComplianceConfig:
    stiffness_min: float = 0.25
    stiffness_contact: float = 1.5
    stiffness_free: float = 5.0
    stiffness_rise_rate: float = 2.0
    stiffness_fall_rate: float = 250.0
    damping_min: float = 0.015
    damping_max: float = 0.2
    damping_rate: float = 2.0
    damping_ratio: float = 1.2
    force_target: float = 2.0
    force_filter_time: float = 0.015
    contact_threshold: float = 0.2
    closure_rate: float = 0.25
    middle_stiffness_multiplier: float = 1.2
    middle_closure_rate_multiplier: float = 1.5
    release_rate: float = 0.4
    max_close_offset: float = 0.25
    max_release_offset: float = 0.15
    tracking_error_limit: float = 0.2
    velocity_softening: float = 1.0

    def __post_init__(self):
        values = np.array(list(asdict(self).values()))
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Compliance parameters must be finite and positive')
        if not self.stiffness_min <= self.stiffness_contact <= self.stiffness_free:
            raise ValueError('Require stiffness_min <= stiffness_contact <= stiffness_free')
        if self.damping_min > self.damping_max:
            raise ValueError('Require damping_min <= damping_max')


class AdaptiveCompliance:
    def __init__(self, reference, lower, upper, effort, inertia, closure_direction,
                 max_reference_velocity=None, config=None,
                 stiffness_multiplier=None, closure_rate_multiplier=None):
        self.config = config or ComplianceConfig()
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 1 or reference.size == 0:
            raise ValueError('Reference must be a nonempty vector')
        self.size = reference.size
        self.lower = self._vector(lower, 'lower').copy()
        self.upper = self._vector(upper, 'upper').copy()
        self.effort = self._vector(effort, 'effort').copy()
        self.inertia = self._vector(inertia, 'inertia').copy()
        self.direction = self._vector(closure_direction, 'closure_direction').copy()
        self.max_reference_velocity = self._vector(
            np.ones(self.size) if max_reference_velocity is None else max_reference_velocity,
            'max_reference_velocity',
        ).copy()
        self.stiffness_multiplier = self._vector(
            np.ones(self.size) if stiffness_multiplier is None else stiffness_multiplier,
            'stiffness_multiplier',
        ).copy()
        self.closure_rate_multiplier = self._vector(
            np.ones(self.size) if closure_rate_multiplier is None else closure_rate_multiplier,
            'closure_rate_multiplier',
        ).copy()
        if (self.lower >= self.upper).any() or (self.effort <= 0).any() or (self.inertia <= 0).any():
            raise ValueError('Invalid limits, effort or inertia')
        if ((self.max_reference_velocity <= 0).any() or
                (self.stiffness_multiplier <= 0).any() or
                (self.closure_rate_multiplier <= 0).any() or
                not np.isin(self.direction, [-1, 0, 1]).all()):
            raise ValueError('Invalid velocity limits or closure directions')
        self.base_reference = np.clip(self._vector(reference, 'reference'), self.lower, self.upper)
        self.reference = self.base_reference.copy()
        self.filtered_force = np.zeros(self.size)
        self.stiffness = np.full(self.size, self.config.stiffness_free)*self.stiffness_multiplier
        self.damping = self._damping(self.stiffness, self.inertia)

    def _vector(self, value, name):
        value = np.asarray(value, dtype=float)
        if value.shape != (self.size,) or not np.isfinite(value).all():
            raise ValueError(f'{name} must be a finite vector of shape ({self.size},)')
        return value

    def _damping(self, stiffness, inertia):
        c = self.config
        return np.clip(2*c.damping_ratio*np.sqrt(stiffness*inertia), c.damping_min, c.damping_max)

    def update(self, position, velocity, contact_force, dt, inertia=None):
        q = self._vector(position, 'position')
        qd = self._vector(velocity, 'velocity')
        force = self._vector(contact_force, 'contact_force')
        mass = self.inertia if inertia is None else self._vector(inertia, 'inertia')
        if not np.isfinite(dt) or not 0 < dt <= 0.1 or (force < 0).any() or (mass <= 0).any():
            raise ValueError('Invalid timestep, contact load or inertia')
        c = self.config
        filtered = self.filtered_force + (-np.expm1(-dt/c.force_filter_time))*(force-self.filtered_force)
        excess = np.maximum(filtered/c.force_target-1, 0.)
        error = self.reference-q
        tracking = np.clip(np.abs(error)/c.tracking_error_limit, 0., 1.)
        contact_gain = (c.stiffness_contact+(c.stiffness_free-c.stiffness_contact)*tracking)/(1+excess**2)
        desired = np.where(filtered > c.contact_threshold, contact_gain, c.stiffness_free)
        desired *= self.stiffness_multiplier
        desired /= 1+(np.abs(qd)/c.velocity_softening)**2
        desired = np.clip(desired, c.stiffness_min*self.stiffness_multiplier,
                          c.stiffness_free*self.stiffness_multiplier)
        stiffness = self.stiffness + np.clip(desired-self.stiffness,
                                             -c.stiffness_fall_rate*self.stiffness_multiplier*dt,
                                             c.stiffness_rise_rate*self.stiffness_multiplier*dt)
        damping_goal = np.clip(self._damping(stiffness, mass)*(1+np.minimum(excess, 3.)),
                               c.damping_min, c.damping_max)
        damping = self.damping + np.clip(damping_goal-self.damping, -c.damping_rate*dt, c.damping_rate*dt)
        # Bounded equilibrium adaptation: flex toward contact, back off overload.
        deficit = (c.force_target-filtered)/c.force_target
        closure_rate = c.closure_rate*self.closure_rate_multiplier
        rate = np.clip(deficit*closure_rate, -c.release_rate, closure_rate)
        saturated = np.abs(self.stiffness*error-self.damping*qd) >= .95*self.effort
        rate = np.where((rate > 0) & ((self.direction*error > c.tracking_error_limit) | saturated), 0., rate)
        speed = self.direction*rate
        speed = np.clip(speed, -self.max_reference_velocity, self.max_reference_velocity)
        reference = self.reference+speed*dt
        a = self.base_reference-self.direction*c.max_release_offset
        b = self.base_reference+self.direction*c.max_close_offset
        reference = np.clip(reference, np.maximum(self.lower, np.minimum(a,b)),
                             np.minimum(self.upper, np.maximum(a,b)))
        self.filtered_force, self.stiffness, self.damping, self.reference = filtered, stiffness, damping, reference
        return {'reference': reference.copy(), 'stiffness': stiffness.copy(),
                'damping': damping.copy(), 'filtered_force': filtered.copy(),
                'torque': np.clip(stiffness*(reference-q)-damping*qd, -self.effort, self.effort)}
