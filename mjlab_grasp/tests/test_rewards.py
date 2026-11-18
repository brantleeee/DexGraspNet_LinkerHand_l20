"""Reward ordering, dt scaling and anti-exploitation checks for v3."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import test_stability
import torch

from mjlab_grasp import rewards
from mjlab_grasp.task import make_env


def fixture(angle=0.0, angular_speed=0.0, contact=True):
    env = test_stability.StabilityTests.make_env(None, angle, angular_speed)
    env.scene["grasp_contact"] = SimpleNamespace(
        data=SimpleNamespace(found=torch.tensor([[contact]]))
    )
    env.scene["hand"].data.joint_vel = torch.zeros(1, 21)
    env.action_manager = SimpleNamespace(
        action=torch.zeros(1, 16), prev_action=torch.zeros(1, 16)
    )
    env.termination_manager = SimpleNamespace(terminated=torch.tensor([False]))
    return env


def total(env):
    return sum(
        term.weight * term.func(env) * env.step_dt
        for term in rewards.make_rewards().values()
    )


class RewardTests(unittest.TestCase):
    def test_speed_limit_is_twenty_percent_of_urdf(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "grasp_generation/mjcf/linkerhand_l20_right.urdf"
        )
        joints = [
            j
            for j in ET.parse(path).getroot().findall("joint")
            if j.find("limit") is not None
        ]
        self.assertEqual(len(joints), 21)
        for joint in joints:
            self.assertAlmostEqual(
                float(joint.find("limit").get("velocity")) * 0.2,
                rewards.JOINT_SPEED_LIMIT,
            )

    def test_speed_penalty_threshold_sign_and_single_joint(self):
        env = fixture()
        for speed in [0.0, 0.1, 0.2, -0.2]:
            env.scene["hand"].data.joint_vel[:] = speed
            self.assertEqual(float(rewards.joint_motion(env)), 0.0)
        env.scene["hand"].data.joint_vel[:] = 0.0
        env.scene["hand"].data.joint_vel[:, -1] = 0.4  # passive thumb DIP
        single = float(rewards.joint_motion(env))
        self.assertGreater(single, 0.0)
        env.scene["hand"].data.joint_vel[:] = -0.4
        self.assertAlmostEqual(float(rewards.joint_motion(env)), single)
        env.scene["hand"].data.joint_vel[:] = 1.0
        self.assertGreater(float(rewards.joint_motion(env)), single)

    def test_fast_finger_cannot_earn_stable_hold(self):
        env = fixture()
        self.assertGreater(float(rewards.stable_hold(env)), 0.0)
        self.assertEqual(float(rewards.strict_hold(env)), 1.0)
        env.scene["hand"].data.joint_vel[:, -1] = 0.21
        self.assertEqual(float(rewards.stable_hold(env)), 0.0)
        self.assertEqual(float(rewards.strict_hold(env)), 0.0)

    def test_stable_beats_rotating_and_falling(self):
        stable = fixture()
        rotating = fixture(angle=0.8, angular_speed=1.0)
        falling = fixture(contact=False)
        falling.scene["object"].data.root_link_pos_w[:, 2] = -0.05
        falling.scene["object"].data.root_link_lin_vel_w[:, 2] = -0.5
        self.assertGreater(float(total(stable)), float(total(rotating)))
        self.assertGreater(float(total(rotating)), float(total(falling)))
        self.assertGreater(float(rewards.stable_hold(stable)), 0.99)
        self.assertLess(float(rewards.stable_hold(rotating)), 0.001)

    def test_no_contact_gets_no_positive_grasp_rewards(self):
        env = fixture(contact=False)
        for func in [
            rewards.capture,
            rewards.orientation,
            rewards.lift,
            rewards.stable_hold,
            rewards.strict_hold,
        ]:
            self.assertEqual(float(func(env)), 0.0)

    def test_drop_is_fixed_event_cost_not_a_dt_scaled_rate(self):
        env = fixture()
        term = rewards.make_rewards()["drop"]
        env.termination_manager.terminated[:] = True
        for dt in [0.002, 0.02, 0.04]:
            env.step_dt = dt
            self.assertAlmostEqual(
                float(term.func(env) * term.weight * dt), -25.0, places=4
            )
        env.termination_manager.terminated[:] = False  # includes ordinary timeouts
        self.assertEqual(float(term.func(env)), 0.0)

    def test_early_drop_cannot_avoid_larger_running_penalties(self):
        cfg = rewards.make_rewards()
        # All six regularizers are in [0, 1]; their worst five-second
        # cumulative penalty is < one fixed failure cost.
        names = [
            "slip",
            "position_error",
            "orientation_error",
            "saturation",
            "action_change",
            "joint_motion",
        ]
        self.assertLess(
            sum(abs(cfg[n].weight) for n in names) * 5.0, abs(cfg["drop"].weight)
        )

    def test_expected_lift_speed_has_zero_linear_slip(self):
        env = fixture()
        env.episode_length_buf[:] = 50  # t=1s, halfway through wrist lift
        env.scene["object"].data.root_link_lin_vel_w[:, 2] = 0.15
        self.assertLess(float(rewards.velocity_error(env)), 1e-6)

    def test_stable_hold_return_increases_with_duration(self):
        env = fixture()
        rate = rewards.make_rewards()["stable_hold"].weight * rewards.stable_hold(env)
        self.assertGreater(float(rate * 2.0), float(rate * 0.2))

    def test_v3_preserves_original_action_and_physics(self):
        old, new = make_env(num_envs=2), make_env(num_envs=2, version="v3")
        self.assertIsInstance(new.rewards, dict)
        self.assertIsInstance(old.rewards, dict)
        self.assertEqual(old.actions, new.actions)
        self.assertEqual(old.sim, new.sim)
        self.assertEqual(old.terminations, new.terminations)
        self.assertEqual(new.episode_length_s, 5.0)


if __name__ == "__main__":
    unittest.main()
