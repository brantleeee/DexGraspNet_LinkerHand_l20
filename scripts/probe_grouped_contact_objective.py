"""Frozen grouped5 objective probes. No optimization and no dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'grasp_generation'))
from diagnose_l20_middle_objective import pose_from_record
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.energy import cal_energy
from utils.contact_library import attach_library, world_contact_normals, normal_alignment
from utils.root_repulsion import filtered_self_repulsion


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output directory')
    if not torch.cuda.is_available():
        p.error('TorchSDF requires CUDA')
    source = a.run.resolve()
    meta = json.loads((source / 'generation_contact_model.json').read_text())
    hand_path = ROOT / 'grasp_generation/utils/hand_model.py'
    expected = meta['code_sha256'].get(str(hand_path))
    if not expected or sha(hand_path) != expected:
        p.error('Hand model differs from recorded run')
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    library = source / 'contact_library'
    hand = HandModel(model_path=str(ROOT / 'grasp_generation/mjcf/linkerhand_l20_right.urdf'),
        mesh_path=str(ROOT / 'grasp_generation/mjcf/meshes'),
        contact_points_path=str(library / 'contact_points.json'),
        penetration_points_path=str(ROOT / 'grasp_generation/mjcf/penetration_points.json'),
        device='cuda', force_mesh_sdf=True)
    attach_library(hand, library)
    obj = ObjectModel(str(ROOT / 'data/meshdata'), 1, 2000, device='cuda')
    obj.initialize(['power-drill-clean-l20'], scales=[1.])
    data_path = source / 'results/scales/scale_1/power-drill-clean-l20.npy'
    data = np.load(data_path, allow_pickle=True)
    owners = [list(hand.mesh)[i] for i in hand.global_index_to_link_index.tolist()]
    middle = [i for i, n in enumerate(owners) if n.startswith('middle_')]
    mcp = 9 + hand.joints_names.index('middle_mcp_pitch')
    pip = 9 + hand.joints_names.index('middle_pip')
    rows = []
    for candidate in np.argsort([r['energy'] for r in data])[:a.top_k]:
        record = data[candidate]
        base, _ = pose_from_record(record, hand, 'cuda')
        ids = torch.tensor([record['contact_point_indices']], device='cuda')
        slot = next(i for i, v in enumerate(ids[0].tolist()) if owners[v].startswith('middle_'))
        for delta in [0., .1, .2, .4]:
            pose = base.clone()
            pose[0, mcp] += delta
            pose = hand.joint_constraints.project(pose).detach().requires_grad_(True)
            hand.set_parameters(pose, ids)
            terms = cal_energy(hand, obj, verbose=True)
            spen = filtered_self_repulsion(hand, record['root_contact_policy'])
            distance, normal = obj.cal_distance(hand.contact_points)
            alignment = normal_alignment(world_contact_normals(hand), normal)
            parts = {'fc': terms[1], 'dis': 100 * terms[2], 'pen': 100 * terms[3],
                     'spen': 10 * spen, 'joints': terms[5], 'normal': meta['w_normal'] * alignment}
            parts['total'] = sum(parts.values())
            gradients = {}
            for name, value in parts.items():
                g = torch.autograd.grad(value.sum(), pose, retain_graph=True, allow_unused=True)[0]
                if g is None:
                    g = torch.zeros_like(pose)
                g = hand.joint_constraints.pullback(g)
                gradients[name] = {'mcp': float(g[0, mcp]), 'pip': float(g[0, pip])}
            with torch.no_grad():
                all_dist, _ = obj.cal_distance(hand.get_contact_candidates())
                nearest = middle[int(all_dist[0, middle].abs().argmin())]
            rows.append(dict(candidate=int(candidate),delta_mcp=delta,mcp_rad=float(pose[0,mcp]),
                pip_rad=float(pose[0,pip]),selected_middle_id=int(ids[0,slot]),
                selected_middle_link=owners[int(ids[0,slot])],
                selected_distance_mm=float(distance[0,slot].abs()*1000),
                selected_signed_distance_mm=float(distance[0,slot]*1000),
                nearest_middle_id=nearest,nearest_middle_link=owners[nearest],
                nearest_middle_distance_mm=float(all_dist[0,nearest].abs()*1000),
                weighted_terms={k:float(v[0]) for k,v in parts.items()},weighted_gradients=gradients))
    a.output.mkdir(parents=True)
    np.save(a.output/'object_surface_points.npy', obj.surface_points_tensor.cpu().numpy())
    provenance = dict(source=str(source),source_result_sha256=sha(data_path),hand_model_sha256=sha(hand_path),
        object_surface_seed=a.seed,
        scope='Frozen poses only. New fixed object surface sample; E_pen is not an exact reproduction of the source run. Positive gradient means descent decreases angle.',
        rows=rows)
    (a.output/'results.json').write_text(json.dumps(provenance,indent=2)+'\n')
    lines=['# grouped5中指完整目标探针','','冻结姿态，无优化/动力学。新固定物体表面采样，E_pen不作源运行精确复现。梯度正值表示下降方向减小关节角。','',
        '|候选|MCP偏移rad|所选连杆|所选点距mm|中指最近点距mm|加权总梯度MCP|加权距离梯度MCP|加权自排斥梯度MCP|',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        g=r['weighted_gradients']
        lines.append(f"|{r['candidate']}|{r['delta_mcp']}|{r['selected_middle_link']}|{r['selected_distance_mm']:.5g}|{r['nearest_middle_distance_mm']:.5g}|{g['total']['mcp']:.5g}|{g['dis']['mcp']:.5g}|{g['spen']['mcp']:.5g}|")
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(a.output/'REPORT.md')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
