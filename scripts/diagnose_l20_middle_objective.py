"""Frozen-pose L20 objective probes, not generation, RL or dynamics validation.

Full contact/penetration gradients use the generation TorchSDF CUDA implementation.
--self-only permits a clearly labelled partial CPU keypoint-repulsion audit.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
import transforms3d

from l20_objective_diagnostics import (contact_probes, contact_terms, finger,
                                       gradient, legacy_pair_loss, pair_masks)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'grasp_generation'))
DEFAULT = ROOT / 'data/experiments/l20_clean_drill_20261115/results/scales/scale_1/power-drill-clean-l20.npy'
URDF = ROOT / 'grasp_generation/mjcf/linkerhand_l20_right.urdf'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def scalar(x):
    return float(x.detach().reshape(-1)[0])


def vector(x, names):
    return {n: float(x[0, 9 + i]) for i, n in enumerate(names)}


def pose_from_record(record, hand, device):
    q = record['qpos']
    r = transforms3d.euler.euler2mat(*[q['WRJR' + k] for k in 'xyz'])
    value = [q['WRJT' + k] for k in 'xyz'] + r[:, :2].T.ravel().tolist() + [q[n] for n in hand.joints_names]
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite source pose; refusing TorchSDF queries')
    pose = torch.tensor([value], dtype=torch.float, device=device)
    projected = hand.joint_constraints.project(pose)
    correction = float((pose - projected).abs().max())
    if correction > 1e-5:
        raise ValueError(f'Source pose violates current URDF: correction={correction} rad/m')
    return projected.detach(), correction


def sampled_pair_geometry(hand, left, right, samples=128):
    """Bidirectional sampled vertex SDF, not certified mesh-mesh intersection."""
    import trimesh
    from torchsdf import compute_sdf
    world = {}
    closed = {}
    for name in [left, right]:
        mesh = hand.mesh[name]
        frame = hand.current_status[name].get_matrix()[0]
        local = mesh['vertices'] @ frame[:3, :3].T + frame[:3, 3]
        world[name] = (local @ hand.global_rotation[0].T + hand.global_translation[0]).detach()
        audited = trimesh.Trimesh(mesh['vertices'].detach().cpu().numpy(),
                                 mesh['faces'].detach().cpu().numpy(), process=True)
        closed[name] = bool(audited.is_watertight and audited.is_winding_consistent)
    rows = []
    for source, target in [(left, right), (right, left)]:
        ids = torch.linspace(0, len(world[source]) - 1, min(samples, len(world[source])), device=world[source].device).long()
        faces = world[target][hand.mesh[target]['faces'].long()]
        distance, sign, _, _ = compute_sdf(world[source][ids].contiguous(), faces.contiguous())
        signed = distance.clamp(min=0).sqrt() * -sign
        rows.append(dict(source=source, target=target, samples=len(ids),
                         min_unsigned_mm=scalar(distance.clamp(min=0).sqrt().min()) * 1000,
                         max_sampled_inside_mm=scalar(signed.clamp(min=0).max()) * 1000 if closed[target] else None,
                         target_closed_winding_consistent=closed[target],
                         note='Sampled vertex proximity only; open/inconsistent target sign is suppressed, and no sampled inside points does not certify no intersection.'))
    return rows


def evaluate(hand, obj, base_pose, probes, masks, owners, weights, self_only,
             detailed=False, geometry_pairs=0):
    from utils.energy import cal_energy
    pose = base_pose.detach().clone().requires_grad_(True)
    first_ids = torch.tensor([probes['global4'][0]], dtype=torch.long, device=pose.device)
    hand.set_parameters(pose, first_ids)
    names = hand.joints_names
    constraints = hand.joint_constraints
    if self_only:
        original_sp = hand.self_penetration()
        base_terms = {}
    else:
        original = cal_energy(hand, obj, verbose=True, **weights)
        original_sp = original[4]
        base_terms = {'E_pen': original[3], 'E_joints': original[5]}
    pairs = legacy_pair_loss(hand.get_penetraion_keypoints()[0])
    losses = {key: (pairs * torch.tensor(mask, device=pose.device)).sum().reshape(1)
              for key, mask in masks.items()}
    raw_sp = sum(losses.values())
    if not torch.allclose(raw_sp, original_sp, atol=2e-5, rtol=1e-5):
        raise RuntimeError('Repulsion decomposition differs from original HandModel')
    filtered_sp = raw_sp - losses['same_link'] - losses['adjacent_links']
    base_grads = {key: gradient(value, pose, constraints) for key, value in base_terms.items()}
    sp_grads = {key: gradient(value, pose, constraints) for key, value in losses.items()}
    g_sp = sum(sp_grads.values())
    g_filtered = g_sp - sp_grads['same_link'] - sp_grads['adjacent_links']
    info = {'terms': {k: scalar(v) for k, v in base_terms.items()},
            'repulsion_categories': {k: dict(value=scalar(v), independent_gradient=vector(sp_grads[k], names)) for k, v in losses.items()},
            'original_repulsion': scalar(original_sp), 'decomposed_repulsion': scalar(raw_sp),
            'filtered_repulsion_probe': scalar(filtered_sp), 'probes': []}
    if detailed:
        terms = {**base_terms, 'E_spen': raw_sp}
        grads = {**base_grads, 'E_spen': g_sp}
        info['base_gradients'] = {k: vector(grads[k], names) for k in terms}
        pair_rows = []
        for i, left in enumerate(sorted(set(owners))):
            for right in sorted(set(owners))[i:]:
                if not (left.startswith('middle_') or right.startswith('middle_')):
                    continue
                mask = np.array([x == left for x in owners])[:, None] & np.array([x == right for x in owners])[None, :]
                if left != right:
                    mask |= mask.T.copy()
                loss = (pairs * torch.tensor(mask, device=pose.device)).sum()
                if scalar(loss) == 0:
                    continue
                g = gradient(loss, pose, constraints)
                pair_rows.append(dict(left=left, right=right, value=scalar(loss),
                                      weighted_mcp_gradient=weights['w_spen'] * float(g[0, 9 + names.index('middle_mcp_pitch')]),
                                      weighted_pip_gradient=weights['w_spen'] * float(g[0, 9 + names.index('middle_pip')])))
        pair_rows.sort(key=lambda r: abs(r['weighted_mcp_gradient']), reverse=True)
        for row in pair_rows[:geometry_pairs]:
            row['sampled_geometry'] = sampled_pair_geometry(hand, row['left'], row['right'])
        info['middle_pair_contributions'] = pair_rows
        info['pair_mcp_gradient_sum_error'] = abs(sum(x['weighted_mcp_gradient'] for x in pair_rows) - weights['w_spen'] * float(g_sp[0, 9 + names.index('middle_mcp_pitch')]))
    if self_only:
        return info
    all_contacts = hand.get_contact_candidates()
    distance, normals = obj.cal_distance(all_contacts)
    info['per_finger_candidate_proximity'] = {}
    candidate_links = [list(hand.mesh)[i] for i in hand.global_index_to_link_index.tolist()]
    for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        mask = torch.tensor([finger(n) == f for n in candidate_links], device=pose.device)
        d = distance[0, mask]
        info['per_finger_candidate_proximity'][f] = dict(min_abs_mm=scalar(d.abs().min()) * 1000,
            within_1mm=int((d.abs() <= .001).sum()), samples=len(d), max_sampled_inside_mm=scalar(d.clamp(min=0).max()) * 1000)
    fixed_gradient = weights['w_pen'] * base_grads['E_pen'] + weights['w_joints'] * base_grads['E_joints']
    for mode, sets in probes.items():
        for repeat, ids in enumerate(sets):
            points = all_contacts[:, ids]
            fc, dis, signed, normal = contact_terms(points, obj)
            g_fc = gradient(fc, pose, constraints)
            g_dis = gradient(dis, pose, constraints)
            for repulsion, sp, gs in [('legacy', raw_sp, g_sp), ('topology_filter_probe', filtered_sp, g_filtered)]:
                total = fc + weights['w_dis'] * dis + weights['w_pen'] * base_terms['E_pen'] + weights['w_joints'] * base_terms['E_joints'] + weights['w_spen'] * sp
                g = g_fc + weights['w_dis'] * g_dis + fixed_gradient + weights['w_spen'] * gs
                # Unit preconditioner probe; the historical optimizer EMA was not saved.
                projected = constraints.project(pose.detach() - .005 * g)
                row = dict(mode=mode, repeat=repeat, repulsion=repulsion, contact_indices=ids,
                           contact_links=[candidate_links[i] for i in ids],
                           points_m=points[0].detach().cpu().tolist(),
                           signed_distance_mm=(signed[0].detach().cpu().numpy() * 1000).tolist(),
                           normals=normal[0].detach().cpu().tolist(),
                           total=scalar(total), E_fc=scalar(fc), E_dis=scalar(dis),
                           E_pen=scalar(base_terms['E_pen']), E_spen=scalar(sp),
                           independent_gradient=vector(g, names),
                           weighted_fc_gradient=vector(g_fc, names),
                           weighted_distance_gradient=vector(weights['w_dis'] * g_dis, names),
                           projected_unit_preconditioner_delta_rad=vector(projected - pose.detach(), names))
                info['probes'].append(row)
                if mode == 'global4' and repeat == 0 and repulsion == 'legacy':
                    direct_grad = gradient(original[0], pose, constraints)
                    grad_error = float((direct_grad - g).abs().max())
                    info['original_energy_crosscheck'] = dict(value_abs_error=abs(scalar(original[0]) - scalar(total)), max_gradient_abs_error=grad_error)
                    if not torch.allclose(direct_grad, g, atol=3e-4, rtol=2e-4) or not torch.allclose(original[0], total, atol=3e-4, rtol=1e-5):
                        raise RuntimeError('Diagnostic contact terms disagree with generation energy/gradient')
    return info


def export_reports(out, results, config):
    import plotly.graph_objects as go
    lines = ['# L20中指接触目标诊断', '',
             f'模式：{"仅自排斥部分检查" if config["self_only"] else "TorchSDF CUDA完整目标探针"}。冻结姿态，无优化、无动力学。', '',
             '旧最终接触索引缺失时，表中均为新保存的受控接触探针，不能重建原接触选择。',
             'topology_filter_probe仅作梯度消融；未证明屏蔽相邻结构在实际碰撞中安全。',
             '梯度正值表示该项的下降方向减小对应关节角；投影步使用单位预条件，非历史RMSProp真实更新。', '',
             '|候选|点对类别|原始损失|加权MCP梯度|加权PIP梯度|', '|---|---|---:|---:|---:|']
    charts = []
    probe_csv, pair_csv, scan_csv = [], [], []
    for r in results:
        candidate = r['candidate']
        base = r['baseline']
        for category, value in base['repulsion_categories'].items():
            g = value['independent_gradient']
            lines.append(f'|{candidate}|{category}|{value["value"]:.6g}|{config["weights"]["w_spen"]*g["middle_mcp_pitch"]:.6g}|{config["weights"]["w_spen"]*g["middle_pip"]:.6g}|')
        for p in base['probes']:
            probe_csv.append(dict(candidate=candidate, mode=p['mode'], repeat=p['repeat'], repulsion=p['repulsion'],
                                  contacts=';'.join(p['contact_links']), total=p['total'], E_fc=p['E_fc'], E_dis=p['E_dis'], E_pen=p['E_pen'], E_spen=p['E_spen'],
                                  middle_mcp_gradient=p['independent_gradient']['middle_mcp_pitch'],
                                  middle_pip_gradient=p['independent_gradient']['middle_pip'],
                                  distance_mcp_gradient=p['weighted_distance_gradient']['middle_mcp_pitch'],
                                  mcp_projected_probe_delta=p['projected_unit_preconditioner_delta_rad']['middle_mcp_pitch']))
        for p in base['middle_pair_contributions']:
            pair_csv.append(dict(candidate=candidate, **{k: v for k, v in p.items() if k != 'sampled_geometry'}))
        fig = go.Figure()
        for mode in ['global4', 'without_middle4', 'grouped4']:
            for repulsion in ['legacy', 'topology_filter_probe']:
                x, y = [], []
                for scan in r['scan']:
                    for p in scan['evaluation']['probes']:
                        if p['mode'] == mode and p['repulsion'] == repulsion:
                            x.append(scan['actual_delta_rad']); y.append(p['total'])
                            scan_csv.append(dict(candidate=candidate, requested_delta_rad=scan['requested_delta_rad'], actual_delta_rad=scan['actual_delta_rad'], mode=mode, repulsion=repulsion, energy=p['total'],
                                                 middle_min_abs_mm=scan['evaluation']['per_finger_candidate_proximity']['middle']['min_abs_mm'],
                                                 middle_sampled_inside_mm=scan['evaluation']['per_finger_candidate_proximity']['middle']['max_sampled_inside_mm']))
                if x:
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'{mode}/{repulsion}'))
        if fig.data:
            fig.update_layout(title=f'候选{candidate}：冻结其他关节的MCP扫描（非优化轨迹）', xaxis_title='MCP实际增量 rad', yaxis_title='对应探针能量')
            charts.append(fig.to_html(full_html=False, include_plotlyjs=True if not charts else False))
            distance_fig = go.Figure()
            for key, label in [('min_abs_mm', '中指接触候选最小绝对距离 mm'), ('max_sampled_inside_mm', '中指接触候选最大内部距离 mm')]:
                distance_fig.add_trace(go.Scatter(x=[s['actual_delta_rad'] for s in r['scan']],
                    y=[s['evaluation']['per_finger_candidate_proximity']['middle'][key] for s in r['scan']], name=label, mode='lines+markers'))
            distance_fig.update_layout(title=f'候选{candidate}：中指接近与内部距离（开放网格符号非认证）', xaxis_title='MCP实际增量 rad', yaxis_title='mm')
            charts.append(distance_fig.to_html(full_html=False, include_plotlyjs=False))
    lines += ['', '## 完整探针均值', '', '|候选|接触模式|排斥模式|平均总梯度MCP|平均接触距离梯度MCP|', '|---|---|---|---:|---:|']
    for candidate in [r['candidate'] for r in results]:
        for mode in ['global4', 'without_middle4', 'grouped4']:
            for repulsion in ['legacy', 'topology_filter_probe']:
                rows = [r for r in probe_csv if r['candidate'] == candidate and r['mode'] == mode and r['repulsion'] == repulsion]
                if rows:
                    lines.append(f'|{candidate}|{mode}|{repulsion}|{np.mean([r["middle_mcp_gradient"] for r in rows]):.6g}|{np.mean([r["distance_mcp_gradient"] for r in rows]):.6g}|')
    lines += ['', '有限差分/逐连杆梯度/采样几何/初始接触点及法向详见results.json；CSV为可读汇总。',
              '生成物体为开放网格，沿用原生成器符号距离；内部/穿透符号不作为独立物理认证。逐连杆采样中，开放/绕向不一致网格不输出内部深度。',
              'E_pen依赖本次固定的物体采样，不能与旧日志中另一份采样直接相减。',
              'TorchSDF法向在原实现中不参与反向传播；此处复现该行为，而不是完整光滑法向模型。',
              '下一步：根据具体连杆对和含/不含中指探针的加权梯度决定修改，不自动推断唯一根因或启动长训。']
    text = '\n'.join(lines) + '\n'
    (out / 'REPORT.md').write_text(text)
    for name, rows in [('contact_probes.csv', probe_csv), ('pair_gradients.csv', pair_csv), ('mcp_scan.csv', scan_csv)]:
        with (out / name).open('w') as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    links = ''.join(f'<p><a href="candidate_{r["candidate"]:03d}_contacts.html">候选{r["candidate"]}静态接触探针（非物理接触）</a></p>' for r in results)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>L20目标诊断</title><pre>' + html.escape(text) + '</pre>' + links + ''.join(charts))


def export_contact_html(hand, obj, pose, probes, candidate, out):
    import plotly.graph_objects as go
    hand.set_parameters(pose)
    traces = hand.get_plotly_data(0, opacity=.4)
    for trace in traces:
        if trace.name.startswith('middle_'): trace.color = '#d62728'
    if obj is not None: traces += obj.get_plotly_data(0, color='gray', opacity=.35)
    fig = go.Figure(traces)
    points = hand.get_contact_candidates()[0].detach().cpu().numpy()
    owners = [list(hand.mesh)[i] for i in hand.global_index_to_link_index.tolist()]
    for mode, sets in probes.items():
        ids = sets[0]; v = points[ids]
        fig.add_trace(go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers',
            marker=dict(size=6), text=[owners[i] for i in ids], name=mode+' repeat0'))
    fig.update_layout(scene_aspectmode='data', title=f'候选{candidate}冻结姿态；中指红；探针点不代表已形成物理接触')
    fig.write_html(out/f'candidate_{candidate:03d}_contacts.html', include_plotlyjs=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-file', type=Path, default=DEFAULT)
    p.add_argument('--object-code', default='power-drill-clean-l20')
    p.add_argument('--candidates', nargs='+', type=int, default=[22, 18, 31], help='NPY原始下标，不是能量排名')
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--surface-samples', type=int, default=2000)
    p.add_argument('--scan-deltas', nargs='+', type=float, default=[0., .015, .05, .1])
    p.add_argument('--fd-epsilon', type=float, default=.001)
    p.add_argument('--geometry-pairs', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--self-only', action='store_true')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.repeats < 1 or a.surface_samples < 1 or not 0 <= a.seed < 2**32 or a.fd_epsilon <= 0 or not np.isfinite(a.fd_epsilon) or not np.isfinite(a.scan_deltas).all() or not 0 <= a.geometry_pairs <= 50:
        p.error('Invalid repeats/samples/scan/finite-difference/geometry count')
    if a.output.exists(): p.error('Output must be a new directory')
    device = torch.device(a.device)
    if not a.self_only and device.type != 'cuda': p.error('Full original TorchSDF gradients require CUDA; CPU only supports --self-only')
    if device.type == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable; run on the GPU host, or explicitly use --device cpu --self-only for partial repulsion only')
    raw = np.load(a.result_file, allow_pickle=True)
    if len(set(a.candidates)) != len(a.candidates) or any(i < 0 or i >= len(raw) for i in a.candidates): p.error('Invalid/duplicate NPY candidate indices')
    scales = [float(raw[i]['scale']) for i in a.candidates]
    if not np.isfinite(scales).all() or min(scales) <= 0 or not np.allclose(scales, scales[0], rtol=0, atol=1e-7): p.error('All candidates must have the same positive saved scale')
    from utils.grasp_results import verify_mesh_identity
    mesh = ROOT / 'data/meshdata' / a.object_code / 'coacd/decomposed.obj'
    if verify_mesh_identity(a.result_file, mesh) is None: p.error('Missing generation mesh identity manifest')
    a.output.mkdir(parents=True)
    weights = dict(w_dis=100., w_pen=100., w_spen=10., w_joints=1.)
    config = {**vars(a), 'weights': weights, 'scale': scales[0], 'self_only': a.self_only,
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'note': 'Frozen-pose controlled probes, no optimization, no dynamics, no reconstruction of unsaved historical contacts.'}
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in config.items()}
    paths = [Path(__file__), ROOT/'scripts/l20_objective_diagnostics.py', URDF, mesh, a.result_file,
             ROOT/'grasp_generation/utils/hand_model.py', ROOT/'grasp_generation/utils/energy.py', ROOT/'grasp_generation/utils/joint_constraints.py',
             ROOT/'grasp_generation/utils/object_model.py', URDF.parent/'contact_points.json', URDF.parent/'penetration_points.json']
    paths += [(URDF.parent / m.get('filename')).resolve() for m in ET.parse(URDF).getroot().findall('link/visual/geometry/mesh')]
    config['sha256'] = {str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    save(a.output / 'config.json', config)
    save(a.output / 'status.json', dict(complete=False, phase='loading', self_only=a.self_only))
    try:
        from utils.hand_model import HandModel
        from utils.object_model import ObjectModel
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        hand = HandModel(model_path=str(URDF), mesh_path=str(URDF.parent/'meshes'), contact_points_path=str(URDF.parent/'contact_points.json'), penetration_points_path=str(URDF.parent/'penetration_points.json'), device=device)
        config.update(joint_names=hand.joints_names, independent_joint_names=hand.joint_constraints.independent_names,
                      object_surface_points_saved_are_unscaled=True)
        save(a.output/'config.json', config)
        obj = None
        if not a.self_only:
            obj = ObjectModel(str(ROOT/'data/meshdata'), 1, a.surface_samples, device=device)
            obj.initialize([a.object_code], scales=[scales[0]])
            np.save(a.output/'object_surface_points.npy', obj.surface_points_tensor.detach().cpu().numpy())
        contact_links = [list(hand.mesh)[i] for i in hand.global_index_to_link_index.tolist()]
        owners = [list(hand.mesh)[i] for i in hand.global_index_to_link_index_penetration.tolist()]
        adjacent = [(j.find('parent').get('link'), j.find('child').get('link')) for j in ET.parse(URDF).getroot().findall('joint')]
        masks = pair_masks(owners, adjacent)
        u, probes = contact_probes(contact_links, a.repeats, a.seed)
        save(a.output/'contact_sets.json', dict(uniform_draws=u.tolist(), probes=probes, candidate_links=contact_links, keypoint_links=owners, adjacency=adjacent))
        results = []
        for candidate in a.candidates:
            print(f'candidate {candidate}: frozen-pose decomposition and contact probes', flush=True)
            pose, correction = pose_from_record(raw[candidate], hand, device)
            baseline = evaluate(hand, obj, pose, probes, masks, owners, weights, a.self_only, detailed=True, geometry_pairs=0 if a.self_only else a.geometry_pairs)
            info = dict(candidate=candidate, source_index=int(raw[candidate].get('source_index', candidate)), scale=scales[0], source_projection_max_correction=correction,
                        source_qpos={k: float(v) for k,v in raw[candidate]['qpos'].items()}, evaluated_pose=pose.detach().cpu().tolist(), baseline=baseline, scan=[])
            for delta in sorted(set([0.] + a.scan_deltas)):
                moved = pose.clone(); index = 9 + hand.joints_names.index('middle_mcp_pitch')
                moved[0, index] += delta; moved = hand.joint_constraints.project(moved)
                ev = evaluate(hand, obj, moved, {k: v[:1] for k, v in probes.items()}, masks, owners, weights, a.self_only)
                info['scan'].append(dict(requested_delta_rad=delta, actual_delta_rad=float(moved[0, index]-pose[0, index]), evaluation=ev))
            # Check the repulsion derivatives separately; no normal derivatives are assumed.
            info['finite_difference_repulsion'] = {}
            for joint in ['middle_mcp_pitch', 'middle_pip']:
                idx = 9 + hand.joints_names.index(joint); side_values = []
                for delta in [-a.fd_epsilon, a.fd_epsilon]:
                    moved = pose.clone(); moved[0, idx] += delta; moved = hand.joint_constraints.project(moved)
                    hand.set_parameters(moved)
                    side_values.append((float(moved[0, idx]), scalar(hand.self_penetration())))
                denominator = side_values[1][0] - side_values[0][0]
                analytic = sum(v['independent_gradient'][joint] for v in baseline['repulsion_categories'].values())
                numeric = (side_values[1][1]-side_values[0][1])/denominator if denominator else None
                info['finite_difference_repulsion'][joint] = dict(analytic=analytic, finite_difference=numeric, absolute_error=abs(numeric-analytic) if numeric is not None else None, endpoints=side_values,
                    note='One-sided at joint bound; piecewise contact changes and float32 cancellation can affect agreement.')
            results.append(info)
            export_contact_html(hand, obj, pose, probes, candidate, a.output)
            save(a.output/'results.json', results)
            save(a.output/'status.json', dict(complete=False, phase='candidate_saved', completed_candidates=len(results), self_only=a.self_only))
        export_reports(a.output, results, config)
        save(a.output/'status.json', dict(complete=True, phase='complete', completed_candidates=len(results), self_only=a.self_only, dynamics_validated=False))
        print(f'Complete: {a.output.resolve()} (diagnostic only)', flush=True)
    except Exception as exc:
        save(a.output/'error.json', dict(type=type(exc).__name__, message=str(exc)))
        save(a.output/'status.json', dict(complete=False, phase='error', self_only=a.self_only))
        raise


if __name__ == '__main__':
    main()
