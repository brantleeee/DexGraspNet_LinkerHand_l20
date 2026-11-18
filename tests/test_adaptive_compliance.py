"""Controller invariants and contact-response regression on a simple plant."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'grasp_generation'))
from utils.adaptive_compliance import AdaptiveCompliance, ComplianceConfig


def make_controller(reference=.7):
    return AdaptiveCompliance([reference], [0.], [1.], [1.], [.001], [1.])


class ComplianceTests(unittest.TestCase):
    def test_middle_finger_gain_and_closure_are_independent(self):
        controller = AdaptiveCompliance(
            [.4, .4], [0., 0.], [1., 1.], [1., 1.], [.001, .001],
            [1., 1.], [1., 1.], stiffness_multiplier=[1., 1.5],
            closure_rate_multiplier=[1., 2.],
        )
        before = controller.reference.copy()
        state = controller.update([.4, .4], [0., 0.], [0., 0.], .001)
        self.assertAlmostEqual(state['stiffness'][1], 1.5*state['stiffness'][0])
        self.assertAlmostEqual(state['reference'][1]-before[1],
                               2*(state['reference'][0]-before[0]))
        self.assertLessEqual(max(abs(state['torque'])), 1.)

    def test_overload_yields_and_obeys_limits(self):
        controller = make_controller()
        previous = controller.stiffness.copy()
        previous_ref = controller.reference.copy()
        for _ in range(1000):
            state = controller.update([.7], [.1], [30.], .001)
            self.assertTrue(np.isfinite(np.concatenate(list(state.values()))).all())
            self.assertLessEqual(abs(state['reference'][0]-previous_ref[0]), .000400001)
            self.assertLessEqual(state['stiffness'][0]-previous[0], .00200001)
            self.assertLessEqual(previous[0]-state['stiffness'][0], .25000001)
            self.assertLessEqual(abs(state['torque'][0]), 1.)
            previous, previous_ref = state['stiffness'], state['reference']
        self.assertLess(state['stiffness'][0], 1.)
        self.assertLess(state['reference'][0], .7)
        self.assertGreaterEqual(state['reference'][0], .7-controller.config.max_release_offset-1e-10)
        self.assertGreaterEqual(state['damping'][0], controller.config.damping_min)
        self.assertLessEqual(state['damping'][0], controller.config.damping_max)

    def test_anti_windup_and_invalid_sensor_state(self):
        controller = make_controller(.9)
        for _ in range(100):
            state = controller.update([0.], [0.], [0.], .001)
        self.assertEqual(state['reference'][0], .9)
        before = [controller.reference.copy(), controller.filtered_force.copy(), controller.stiffness.copy()]
        with self.assertRaises(ValueError):
            controller.update([0.], [0.], [float('nan')], .001)
        for a,b in zip(before, [controller.reference,controller.filtered_force,controller.stiffness]):
            np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):
            ComplianceConfig(stiffness_min=6.)

    def test_contact_force_reduction_on_a_compliant_wall(self):
        # Rotary inertia drives a 5 cm fingertip against a 1000 N/m surface.
        # This is a controller regression fixture, not a cup stability result.
        def simulate(adaptive):
            controller = make_controller(.8)
            q, velocity, dt = .45, 0., .0002
            forces = []
            for _ in range(10000):
                force = max(0., 1000*.05*(q-.5))
                if adaptive:
                    torque = controller.update([q], [velocity], [force], dt)['torque'][0]
                else:
                    torque = np.clip(5*(.8-q)-.05*velocity, -1, 1)
                velocity += dt*(torque-.05*force-.02*velocity)/.001
                q += dt*velocity
                forces.append(force)
            return np.mean(forces[-2500:])
        fixed, adaptive = simulate(False), simulate(True)
        self.assertAlmostEqual(fixed, 10., delta=.1)
        self.assertLess(adaptive, fixed*.6)

    def test_passive_joint_bounds_restrict_the_drive(self):
        from utils.mujoco_compliance import independent_limits, independent_velocity
        robot = ET.fromstring('''<robot>
          <joint name="drive"><limit lower="0" upper="1" velocity="0.8"/></joint>
          <joint name="passive"><limit lower="0" upper="0.5" velocity="0.2"/>
            <mimic joint="drive" multiplier="2"/>
          </joint></robot>''')
        self.assertEqual(independent_limits(robot, 'drive'), (0., .25))
        self.assertEqual(independent_velocity(robot, 'drive'), .1)
        controller = AdaptiveCompliance([0.], [0.], [.25], [1.], [.001], [1.], [.1])
        previous = 0.
        for _ in range(1000):
            state = controller.update([previous], [0.], [0.], .001)
            passive_velocity = 2*(state['reference'][0]-previous)/.001
            self.assertLessEqual(passive_velocity, .2+1e-10)
            self.assertLessEqual(2*state['reference'][0], .5+1e-10)
            previous = state['reference'][0]


if __name__ == '__main__':
    unittest.main()
