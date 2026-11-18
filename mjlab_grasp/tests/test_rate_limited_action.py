"""Check the v5 physical target slew limit and per-environment reset."""

import unittest
from types import SimpleNamespace

import torch

from mjlab_grasp.task import RateLimitedResidualAction, target_state


class RateLimitedActionTests(unittest.TestCase):
    def make_term(self):
        term = object.__new__(RateLimitedResidualAction)
        term._offset = torch.full((2, 2), 0.5)
        term._scale = torch.full((2, 2), 0.5)
        term._raw_actions = torch.zeros(2, 2)
        term._processed_actions = term._offset.clone()
        term._filtered_target = term._offset.clone()
        term._target_step = torch.zeros(2, 2)
        term.cfg = SimpleNamespace(clip=None)
        term._env = SimpleNamespace(
            step_dt=0.02, episode_length_buf=torch.tensor([0, 100])
        )
        return term

    def test_capture_and_hold_limits_and_reset(self):
        term = self.make_term()
        term.process_actions(torch.ones(2, 2))
        self.assertLessEqual(float(term._target_step[0].abs().max()), 0.016001)
        self.assertLessEqual(float(term._target_step[1].abs().max()), 0.004001)
        self.assertGreater(float(term._target_step[0].abs().max()), 0.004)
        for _ in range(10):
            previous = term._processed_actions.clone()
            term.process_actions(-torch.ones(2, 2))
            self.assertLessEqual(float((term._processed_actions[0] - previous[0]).abs().max()), 0.016001)
            self.assertLessEqual(float((term._processed_actions[1] - previous[1]).abs().max()), 0.004001)
        previous_other = term._processed_actions[1].clone()
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term._processed_actions[0], term._offset[0])
        torch.testing.assert_close(term._processed_actions[1], previous_other)
        torch.testing.assert_close(term._target_step[0], torch.zeros(2))

    def test_target_state_reports_applied_command(self):
        term = self.make_term()
        term.process_actions(torch.ones(2, 2))
        env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _: term))
        torch.testing.assert_close(
            target_state(env),
            (term._processed_actions - term._offset) / term._scale,
        )


if __name__ == "__main__":
    unittest.main()
