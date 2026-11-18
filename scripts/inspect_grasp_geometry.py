"""Measure saved grasp contact candidates against the actual scaled object mesh."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import transforms3d
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_file', type=Path)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    result_file = args.result_file.resolve()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root / 'grasp_generation')
    sys.path.insert(0, str(root / 'grasp_generation'))
    from utils.hand_model import HandModel

    hand = HandModel(
        model_path='mjcf/linkerhand_l20_right.urdf', mesh_path='mjcf/meshes',
        contact_points_path='mjcf/contact_points.json',
        penetration_points_path='mjcf/penetration_points.json', device='cpu',
    )
    grasp = np.load(result_file, allow_pickle=True)[args.index]
    q = grasp['qpos']
    rotation = np.array(transforms3d.euler.euler2mat(*[q[f'WRJR{a}'] for a in 'xyz']))
    pose = torch.tensor(
        [q[f'WRJT{a}'] for a in 'xyz'] + rotation[:, :2].T.ravel().tolist()
        + [q[name] for name in hand.joints_names], dtype=torch.float,
    ).unsqueeze(0)
    hand.set_parameters(pose)
    points = hand.get_contact_candidates()[0].detach().numpy()
    mesh = trimesh.load(
        root / 'data/meshdata' / result_file.stem / 'coacd/decomposed.obj',
        force='mesh', process=False,
    )
    mesh.vertices *= float(grasp['scale'])
    _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    assert np.isfinite(distances).all()
    link_names = list(hand.mesh)
    point_links = [link_names[i] for i in hand.global_index_to_link_index.tolist()]
    by_finger = {}
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        mask = np.array([name.startswith(finger + '_') for name in point_links])
        if mask.any():
            by_finger[finger] = {
                'nearest_candidate_distance_mm': float(distances[mask].min() * 1000),
                'candidate_points_within_2mm': int((distances[mask] < 0.002).sum()),
            }
    report = {
        'candidate_index': args.index,
        'scale': float(grasp['scale']),
        'energy_terms': {k: float(grasp[k]) for k in ['energy', 'E_fc', 'E_dis', 'E_pen', 'E_spen', 'E_joints']},
        'nearest_contact_candidates_by_finger': by_finger,
        'note': 'Distances inspect all predefined contact candidates, not the four sampled optimization contacts. Geometric proximity does not establish simulated grasp stability.',
    }
    output = result_file.with_name(f'{result_file.stem}_geometry_{args.index}.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
