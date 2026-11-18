"""The thumb policy and reset state must not depend on saved static thumb angles."""

import copy
import json
import unittest

from mjlab_grasp.task import DEFAULT_REFERENCE, free_thumb_parameters, make_env


class FreeThumbTests(unittest.TestCase):
    def setUp(self):
        self.ref = json.loads((DEFAULT_REFERENCE / "reference.json").read_text())
        self.xml = DEFAULT_REFERENCE / "hand.xml"

    def test_changing_static_thumb_angles_changes_neither_mapping_nor_reset(self):
        changed = copy.deepcopy(self.ref)
        for name in changed["joint_pos"]:
            if name.startswith("thumb_"):
                changed["joint_pos"][name] += 0.37
        self.assertEqual(
            free_thumb_parameters(self.ref, self.xml),
            free_thumb_parameters(changed, self.xml),
        )

    def test_thumb_spans_feasible_limits_other_fingers_keep_static_zero(self):
        initial, offsets, scales = free_thumb_parameters(self.ref, self.xml)
        thumb_count = 0
        for name in self.ref["drive_names"]:
            if name.startswith("thumb_"):
                thumb_count += 1
                low, high = self.ref["drive_limits"][name]
                self.assertAlmostEqual(offsets[name] - scales[name], low)
                self.assertAlmostEqual(offsets[name] + scales[name], high)
                self.assertAlmostEqual(initial[name], (low + high) / 2)
            else:
                self.assertEqual(offsets[name], self.ref["joint_pos"][name])
                self.assertEqual(scales[name], 0.2)
                self.assertEqual(initial[name], self.ref["joint_pos"][name])
        self.assertEqual(thumb_count, 4)
        self.assertAlmostEqual(initial["thumb_dip"], initial["thumb_mcp"] * 1.1619)

    def test_v4_uses_same_slow_speed_rewards_and_physics(self):
        old, new = make_env(version="v3"), make_env(version="v4")
        self.assertEqual(old.rewards, new.rewards)
        self.assertEqual(old.sim, new.sim)
        self.assertFalse(new.actions["residual"].use_default_offset)
        self.assertEqual(len(new.actions["residual"].offset), 16)
        self.assertEqual(new.episode_length_s, 5.0)


if __name__ == "__main__":
    unittest.main()
