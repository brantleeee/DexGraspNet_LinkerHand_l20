from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from l20_objective_diagnostics import contact_probes, legacy_pair_loss, pair_masks


def test_contact_probes_cover_middle_and_reuse_seed_without_changing_point_count():
    links = ['thumb_distal', 'index_distal', 'middle_distal', 'ring_distal', 'pinky_distal']
    u, probes = contact_probes(links, 32, 9)
    other_u, other = contact_probes(links, 32, 9)
    assert np.array_equal(u, other_u) and probes == other
    assert all(len(ids) == 4 for mode in probes.values() for ids in mode)
    assert all(2 not in ids for ids in probes['without_middle4'])
    assert all(ids[:3] == [0, 1, 2] and ids[3] in [3, 4] for ids in probes['grouped4'])


def test_masks_partition_all_ordered_pairs_and_keep_nonadjacent_finger_pairs():
    owners = ['base_link', 'middle_proximal', 'middle_proximal', 'middle_middle', 'middle_distal', 'index_distal']
    masks = pair_masks(owners, [('middle_proximal', 'middle_middle'), ('middle_middle', 'middle_distal')])
    assert np.all(sum(v.astype(int) for v in masks.values()) == 1)
    assert all(np.array_equal(v, v.T) for v in masks.values())
    assert masks['same_link'][1, 2]
    assert masks['adjacent_links'][1, 3]
    assert masks['same_finger_nonadjacent'][1, 4]
    assert masks['different_fingers'][4, 5]
    assert masks['finger_palm'][0, 4]


def test_same_body_repulsion_is_rotation_invariant_but_cross_body_can_push():
    angle = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    c, s = angle.cos(), angle.sin()
    rotation = torch.stack([c, -s, s, c]).reshape(2, 2)
    xy = torch.tensor([[.03, 0.], [.03, .005]], dtype=torch.float64) @ rotation.T
    rotated = torch.cat([xy, torch.zeros(2, 1, dtype=torch.float64)], dim=1)
    fixed = torch.tensor([[.031, .011, 0.]], dtype=torch.float64)
    points = torch.cat([rotated, fixed])
    pairs = legacy_pair_loss(points)
    same = pairs[:2, :2].sum()
    cross = pairs[:2, 2].sum() + pairs[2, :2].sum()
    same_gradient = torch.autograd.grad(same, angle, retain_graph=True)[0]
    cross_gradient = torch.autograd.grad(cross, angle)[0]
    assert abs(float(same_gradient)) < 1e-10
    assert abs(float(cross_gradient)) > 1e-4
    assert torch.allclose(pairs.sum(), same + cross)
