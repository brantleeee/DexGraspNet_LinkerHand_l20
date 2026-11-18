"""Regression counterexample: one-sided surface tests miss nested solids."""
import itertools
import torch


def test_object_surface_only_misses_a_finger_fully_inside_object():
    object_vertices=torch.tensor(list(itertools.product([-1.,1.],repeat=3)))
    finger_vertices=torch.tensor(list(itertools.product([-.1,.1],repeat=3)))
    # Exact positive-inside box fields at these vertices.
    object_points_inside_finger=(.1-object_vertices.abs().amax(-1)).clamp_min(0).sum()
    finger_points_inside_object=(1-finger_vertices.abs().amax(-1)).clamp_min(0).sum()
    assert object_points_inside_finger==0
    assert finger_points_inside_object>7
