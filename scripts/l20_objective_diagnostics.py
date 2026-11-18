"""Diagnostic-only contact probes and legacy keypoint-loss decomposition."""
import numpy as np
import torch


def finger(link):
    return 'palm' if link == 'base_link' else link.split('_')[0]


def pair_masks(owners, adjacent):
    """Disjoint ordered-pair partition; adjacency is diagnostic, not permission."""
    adjacent = {frozenset(p) for p in adjacent}
    labels = ['same_link', 'adjacent_links', 'same_finger_nonadjacent',
              'finger_palm', 'different_fingers']
    masks = {key: np.zeros((len(owners), len(owners)), dtype=bool) for key in labels}
    for i, a in enumerate(owners):
        for j, b in enumerate(owners):
            if a == b:
                key = 'same_link'
            elif frozenset((a, b)) in adjacent:
                key = 'adjacent_links'
            elif finger(a) == finger(b):
                key = 'same_finger_nonadjacent'
            elif 'palm' in (finger(a), finger(b)):
                key = 'finger_palm'
            else:
                key = 'different_fingers'
            masks[key][i, j] = True
    return masks


def legacy_pair_loss(points):
    # Match HandModel.self_penetration, including ordered pairs and coincident rule.
    distance = (points[:, None] - points[None, :] + 1e-13).square().sum(-1).sqrt()
    distance = torch.where(distance < 1e-6, torch.full_like(distance, 1e6), distance)
    return (.02 - distance).clamp(min=0)


def contact_probes(candidate_links, repeats, seed):
    """Shared uniform draws mapped to three explicitly saved selection schemes."""
    rng = np.random.default_rng(seed)
    u = rng.random((repeats, 4))
    all_ids = np.arange(len(candidate_links))
    no_middle = np.array([i for i, n in enumerate(candidate_links) if finger(n) != 'middle'])
    groups = [[i for i, n in enumerate(candidate_links) if finger(n) in names]
              for names in [('thumb',), ('index',), ('middle',), ('ring', 'pinky')]]
    if not len(no_middle) or any(not g for g in groups):
        raise ValueError('L20 contact groups are missing')
    result = {}
    for name in ['global4', 'without_middle4', 'grouped4']:
        samples = []
        for row in u:
            pools = groups if name == 'grouped4' else [all_ids if name == 'global4' else no_middle] * 4
            samples.append([int(pool[min(int(x * len(pool)), len(pool) - 1)])
                            for x, pool in zip(row, pools)])
        result[name] = samples
    return u, result


def contact_terms(points, obj):
    """Exact algebra from energy.cal_energy, whose gradients we also cross-check."""
    distance, normal = obj.cal_distance(points)
    count = points.shape[1]
    matrix = points.new_tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                [0, -1, 0, 1, 0, 0, 0, 0, 0]])
    g = torch.cat([torch.eye(3, device=points.device).expand(1, count, 3, 3).reshape(1, 3 * count, 3),
                   (points @ matrix).reshape(1, 3 * count, 3)], dim=2).float()
    fc = torch.norm(normal.reshape(1, 1, 3 * count) @ g, dim=(1, 2)).square()
    return fc, distance.abs().sum(-1, dtype=torch.float), distance, normal


def gradient(value, pose, constraints):
    if not value.requires_grad:
        return torch.zeros_like(pose)
    raw = torch.autograd.grad(value.sum(), pose, retain_graph=True, allow_unused=True)[0]
    return constraints.pullback(raw if raw is not None else torch.zeros_like(pose))
