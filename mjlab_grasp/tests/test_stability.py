"""CPU checks for residual filtering and stability objectives (mjlab environment)."""

import unittest
from types import SimpleNamespace

import torch

from mjlab_grasp.task import SmoothResidualAction, stable_pose_reward, stable_success


class StabilityTests(unittest.TestCase):
    def test_slew_limit_and_partial_reset(self):
        term = object.__new__(SmoothResidualAction)
        term._offset = torch.full((2, 3), 0.5)
        term._scale = 0.2
        term._raw_actions = torch.zeros(2, 3)
        term._processed_actions = term._offset.clone()
        term._filtered_target = term._offset.clone()
        term._max_delta = 0.02
        term.cfg = SimpleNamespace(clip=None)
        for _ in range(40):
            previous = term._processed_actions.clone()
            term.process_actions(torch.full((2, 3), 100.0))
            self.assertLessEqual(
                float((term._processed_actions - previous).abs().max()), 0.020001
            )
            self.assertLessEqual(
                float((term._processed_actions - term._offset).abs().max()), 0.200001
            )
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term._processed_actions[0], term._offset[0])
        torch.testing.assert_close(term._processed_actions[1], torch.full((3,), 0.7))
        term.reset()
        term.process_actions(torch.zeros(2, 3))
        torch.testing.assert_close(term._processed_actions, term._offset)

    def make_env(self, angle=0.0, angular_speed=0.0):
        obj = SimpleNamespace(
            data=SimpleNamespace(
                default_root_state=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                root_link_pos_w=torch.tensor([[0.0, 0.0, 0.1]]),
                root_link_quat_w=torch.tensor(
                    [
                        [
                            torch.cos(torch.tensor(angle / 2)),
                            0.0,
                            0.0,
                            torch.sin(torch.tensor(angle / 2)),
                        ]
                    ]
                ),
                root_link_lin_vel_w=torch.zeros(1, 3),
                root_link_ang_vel_w=torch.tensor([[0.0, 0.0, angular_speed]]),
            )
        )
        hand = SimpleNamespace(
            data=SimpleNamespace(
                default_root_state=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                root_link_pos_w=torch.tensor([[0.0, 0.0, 0.1]]),
            )
        )

        class Scene(dict):
            env_origins = torch.zeros(1, 3)

        return SimpleNamespace(
            scene=Scene(hand=hand, object=obj),
            episode_length_buf=torch.tensor([100]),
            step_dt=0.02,
        )

    def test_rotation_cannot_get_full_hold_reward(self):
        self.assertAlmostEqual(float(stable_pose_reward(self.make_env())), 1.0)
        self.assertLess(float(stable_pose_reward(self.make_env(angle=0.8))), 0.01)

    def test_success_requires_low_angular_speed(self):
        self.assertEqual(float(stable_success(self.make_env())), 1.0)
        self.assertEqual(float(stable_success(self.make_env(angular_speed=1.0))), 0.0)
        self.assertEqual(float(stable_success(self.make_env(angle=0.8))), 0.0)


if __name__ == "__main__":
    unittest.main()
