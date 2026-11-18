"""Prepare existing meshes and the official DexGraspNet contact adjustment for Gym."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
import transforms3d
import trimesh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/DexGraspNet_official_compat/grasp_generation"
sys.path.insert(0, str(SOURCE))
from utils.hand_model import HandModel
from utils.object_model import ObjectModel

JOINTS = ['robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
          'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
          'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
          'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
          'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0']
OBJECTS = ('core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03', '2_of_Jenga_Classic_Game')


def export_urdf(mesh_path, folder):
    """Use existing connected convex pieces; no new convex decomposition."""
    folder.mkdir(parents=True)
    pieces = list(trimesh.load(mesh_path, force="mesh", process=False).split(only_watertight=False))
    robot = ET.Element("robot", name="coacd")
    for index, piece in enumerate(pieces):
        name = f"convex_{index:03d}"
        piece.export(folder / f"{name}.obj")
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        inertia = piece.moment_inertia
        ET.SubElement(inertial, "inertia", **{key: str(inertia[i, j]) for key, i, j in
                      [('ixx', 0, 0), ('ixy', 0, 1), ('ixz', 0, 2),
                       ('iyy', 1, 1), ('iyz', 1, 2), ('izz', 2, 2)]})
        # The upstream exporter also omits mass; Gym derives it from asset density.
        for kind in ("visual", "collision"):
            geom = ET.SubElement(link, kind)
            ET.SubElement(geom, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ET.SubElement(geom, "geometry"), "mesh", filename=f"{name}.obj", scale="1 1 1")
        if index:
            joint = ET.SubElement(robot, "joint", name=f"fixed_{index}", type="fixed")
            ET.SubElement(joint, "parent", link=f"convex_{index - 1:03d}")
            ET.SubElement(joint, "child", link=name)
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.ElementTree(robot).write(folder / "coacd.urdf")
    return len(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "external/DexGraspNet_official_compat/data/experiments/shadow_cup_jenga_006_20261102/results")
    args = parser.parse_args()
    output, results = args.output.resolve(), args.results.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(SOURCE)
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    manifest = {"source_commit": "bd1b13d7248729af117e1d46aaa6266b147a3c7b",
                "contact_adjustment": {"thres_cont": 0.001, "dis_move": 0.001, "grad_move": 500},
                "objects": {}}
    mesh_root = SOURCE.parent / "data/meshdata"
    for obj in OBJECTS:
        result_path = results / f"{obj}.npy"
        records = np.load(result_path, allow_pickle=True)
        poses, translations, rotations, raw = [], [], [], []
        for record in records:
            q = record["qpos"]
            translation = [q[n] for n in ('WRJTx', 'WRJTy', 'WRJTz')]
            euler = [q[n] for n in ('WRJRx', 'WRJRy', 'WRJRz')]
            rotation = transforms3d.euler.euler2mat(*euler)[:, :2].T.ravel().tolist()
            joints = [q[n] for n in JOINTS]
            poses.append(translation + rotation + joints)
            raw.append(joints)
            translations.append(translation)
            rotations.append(transforms3d.euler.euler2quat(*euler))
        state = torch.tensor(poses, dtype=torch.float, device=device, requires_grad=True)
        hand = HandModel(mjcf_path='mjcf/shadow_hand_wrist_free.xml', mesh_path='mjcf/meshes',
                         contact_points_path='mjcf/contact_points.json',
                         penetration_points_path='mjcf/penetration_points.json',
                         n_surface_points=2000, device=device)
        hand.set_parameters(state)
        model = ObjectModel(data_root_path=str(mesh_root), batch_size_each=len(records), num_samples=0, device=device)
        model.initialize(obj)
        model.object_scale_tensor = torch.tensor([r['scale'] for r in records], device=device).reshape(1, -1)
        points = torch.zeros((len(records), 19, 3), device=device)
        normals = torch.zeros_like(points)
        for i, link in enumerate(hand.mesh):
            if len(hand.mesh[link]['surface_points']) == 0:
                continue
            surface = hand.current_status[link].transform_points(hand.mesh[link]['surface_points']).expand(len(records), -1, 3)
            surface = surface @ hand.global_rotation.transpose(1, 2) + hand.global_translation.unsqueeze(1)
            distance, normal = model.cal_distance(surface)
            nearest = distance.argmax(dim=1)
            near_distance = torch.gather(distance, 1, nearest.unsqueeze(1))
            gather = nearest.reshape(-1, 1, 1).expand(-1, 1, 3)
            admitted = (-near_distance < 0.001).reshape(-1, 1, 1).expand(-1, 1, 3)
            points[:, i:i + 1] = torch.where(admitted, torch.gather(surface, 1, gather), points[:, i:i + 1])
            normals[:, i:i + 1] = torch.where(admitted, torch.gather(normal, 1, gather), normals[:, i:i + 1])
        loss = ((points + normals * 0.001).detach().clone() - points).square().sum()
        loss.backward()
        adjusted = (state[:, 9:] + state.grad[:, 9:] * 500).detach().cpu().numpy()
        np.savez(output / f"{obj}.npz", raw_qpos=np.asarray(raw), force_qpos=adjusted,
                 translations=translations, rotations=rotations,
                 scales=[r['scale'] for r in records], E_pen=[r['E_pen'] for r in records])
        mesh_path = mesh_root / obj / "coacd/decomposed.obj"
        pieces = export_urdf(mesh_path, output / "assets" / obj)
        manifest['objects'][obj] = {"count": len(records), "convex_pieces": pieces,
            "grasp_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            "max_joint_adjustment_rad": float(np.max(np.abs(adjusted - raw)))}
        print(obj, manifest['objects'][obj], flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == '__main__':
    main()
