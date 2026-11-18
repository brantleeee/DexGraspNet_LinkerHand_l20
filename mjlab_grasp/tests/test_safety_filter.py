"""Existing v4 policies keep their capture commands while hold targets are bounded."""

import unittest
from types import SimpleNamespace

import torch

from mjlab_grasp.safety_filter import HoldTargetFilter


class SafetyFilterTests(unittest.TestCase):
    def test_capture_is_unchanged_and_hold_is_limited(self):
        env = SimpleNamespace(step_dt=0.02, episode_length_buf=torch.tensor([25, 100]))
        term = SimpleNamespace(
            _offset=torch.full((2, 1), 0.5),
            _scale=torch.full((2, 1), 0.5),
            _clip=torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]]),
        )
        filt = HoldTargetFilter(env, term, 0.2)
        output = filt(torch.ones(2, 1))
        self.assertAlmostEqual(float(output[0, 0]), 1.0)
        self.assertAlmostEqual(float(output[1, 0]), 0.008)
        env.episode_length_buf[0] = 100
        previous = filt.target.clone()
        filt(-torch.ones(2, 1))
        self.assertLessEqual(float((filt.target - previous).abs().max()), 0.004001)
        filt.reset(torch.tensor([0]))
        self.assertAlmostEqual(float(filt.target[0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
