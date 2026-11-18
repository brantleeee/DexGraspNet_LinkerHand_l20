"""Frozen-pose energy attribution and paired middle-slot enumeration.
Reuses a hash-verified static grid; never runs dynamics or edits source poses.
"""
import argparse,csv,hashlib,html,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'grasp_generation'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balance_terms(points,normals):
    force=normals.sum(-2).square().sum(-1)
    torque=torch.cross(points,normals,dim=-1).sum(-2).square().sum(-1)
    return force,torque


def choose(records,key):
    eligible=[r for r in records if r['admissible'] and r[key] is not None]
    return min(eligible,key=lambda r:r[key]) if eligible else None


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'data/experiments/l20_mesh_gap_compare_20261116_132642')
    p.add_argument('--diagnostic-rows',nargs='+',type=int,default=[24,20,45,43])
    p.add_argument('--diagnostics-only',action='store_true')
    p.add_argument('--region-labels',type=Path,help='Optional semantic annotations, not used to filter A/B')
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new output')
    if not torch.cuda.is_available():p.error('Original TorchSDF evaluation requires CUDA')
    source=a.run.resolve();cfg=json.loads((source/'config.json').read_text());rows=json.loads((source/'results.json').read_text())
    status=json.loads((source/'status.json').read_text())
    if not status['complete']:p.error('Source static comparison incomplete')
    for path,h in cfg['sha256'].items():
        if sha(path)!=h:p.error('Source code/model changed: '+path)
    if any(r['row_index']!=i for i,r in enumerate(rows)):p.error('Source row indices inconsistent')
    if any(i<0 or i>=len(rows) for i in a.diagnostic_rows):p.error('Invalid diagnostic row')
    from diagnose_l20_middle_objective import pose_from_record,URDF
    from utils.hand_model import HandModel
    from utils.object_model import ObjectModel
    from l20_objective_diagnostics import contact_terms
    from utils.grasp_results import verify_mesh_identity
    raw=np.load(cfg['result_file'],allow_pickle=True)
    mesh=ROOT/'data/meshdata/power-drill-clean-l20/coacd/decomposed.obj'
    if verify_mesh_identity(cfg['result_file'],mesh) is None:p.error('Missing source mesh identity')
    hand=HandModel(model_path=str(URDF),mesh_path=str(URDF.parent/'meshes'),contact_points_path=str(URDF.parent/'contact_points.json'),penetration_points_path=str(URDF.parent/'penetration_points.json'),device='cuda')
    obj=ObjectModel(str(ROOT/'data/meshdata'),1,0,device='cuda');obj.initialize(['power-drill-clean-l20'],scales=[1.])
    samples=np.load(source/'object_surface_points.npy')
    if samples.shape!=(1,cfg['surface_samples'],3):p.error('Surface point shape mismatch')
    obj.surface_points_tensor=torch.tensor(samples,device='cuda')
    names=hand.joints_names;owners=[list(hand.mesh)[i] for i in hand.global_index_to_link_index.tolist()]
    middle=[i for i,n in enumerate(owners) if n.startswith('middle_')]
    fixed=cfg['contact_ids'];slot=next(i for i,k in enumerate(fixed) if owners[k].startswith('middle_'))
    if len(middle)!=80 or len(fixed)!=4:p.error('Expected 80 middle candidates and 4 slots')
    labels={}
    if a.region_labels:
        annotation=json.loads(a.region_labels.read_text())
        if annotation['contact_points_sha256']!=sha(URDF.parent/'contact_points.json'):p.error('Region annotations do not match contact candidates')
        labels=annotation['labels']
        if any(v not in ['palmar','dorsal','joint_shell','unknown'] for v in labels.values()):p.error('Invalid region label')
    a.output.mkdir(parents=True)
    provenance=dict(source=str(source),source_files_sha256={n:sha(source/n) for n in ['config.json','results.json','selected_references.json','object_surface_points.npy']},
        implementation_sha256={str(f):sha(f) for f in [Path(__file__),ROOT/'grasp_generation/utils/energy.py',ROOT/'grasp_generation/utils/object_model.py',ROOT/'scripts/l20_objective_diagnostics.py']},
        fixed_contacts=fixed,middle_slot=slot,diagnostic_rows=a.diagnostic_rows,diagnostics_only=a.diagnostics_only,region_labels=labels,
        scope='Same geometry feasible set; middle index only changes. Annotations do not alter A/B selection. No physical contact or dynamics certification.')
    save(a.output/'config.json',provenance);save(a.output/'status.json',dict(complete=False,phase='enumeration'))
    all_pose=[];flat=[];choices=[];poses={}
    active=[r for r in rows if not a.diagnostics_only or r['row_index'] in a.diagnostic_rows]
    for row in active:
        pose,_=pose_from_record(raw[row['candidate']],hand,'cuda')
        pose[0,9+names.index('middle_mcp_pitch')]=row['mcp_rad'];pose[0,9+names.index('middle_pip')]=row['pip_rad'];pose=hand.joint_constraints.project(pose)
        hand.set_parameters(pose,torch.tensor([fixed],device='cuda'))
        with torch.no_grad():
            points=hand.get_contact_candidates();distance,normal=obj.cal_distance(points)
            pen=hand.cal_distance(obj.surface_points_tensor).clamp(min=0)
            if abs(float(pen.sum())-row['E_pen'])>2e-6 or abs(float(pen.max())-row['max_sampled_pen_m'])>2e-6:
                raise ValueError('Cached penetration does not reproduce at row '+str(row['row_index']))
            indices=torch.tensor([fixed[:slot]+[m]+fixed[slot+1:] for m in middle],device='cuda')
            point=points[0][indices];norm=normal[0][indices];signed=distance[0][indices]
            force,torque=balance_terms(point,norm);fc=force+torque;dis=signed.abs().sum(-1)
            gpen=float(pen.sum())*100
            gap=None if row['gap_loss_m'] is None else row['gap_loss_m']*10
            totals=None if gap is None else fc+100*dis+gpen+gap
            where=middle.index(fixed[slot]);original_fc,original_dis,_,_=contact_terms(points[:,fixed],obj)
            if not torch.allclose(fc[where],original_fc[0],atol=2e-5,rtol=1e-5):raise ValueError('Force/moment decomposition mismatch')
            if totals is not None and abs(float(totals[where])-row['mesh_gap_energy'])>5e-4:raise ValueError('Fixed target does not reproduce')
        nearest=middle[int(distance[0,middle].abs().argmin())]
        enumeration=[]
        for j,m in enumerate(middle):
            item=dict(row_index=row['row_index'],candidate=row['candidate'],middle_contact=m,middle_link=owners[m],region_label=labels.get(str(m),'unknown'),
                region_verified=False,selected_middle_distance_mm=float(distance[0,m].abs()*1000),signed_middle_distance_mm=float(distance[0,m]*1000),
                E_fc_force=float(force[j]),E_fc_torque=float(torque[j]),E_fc=float(fc[j]),E_dis=float(dis[j]),weighted_dis=float(100*dis[j]),
                weighted_pen=gpen,weighted_gap=gap,total=None if totals is None else float(totals[j]),admissible=row['admissible'],source_rejection=row['reason'])
            enumeration.append(item);flat.append(item)
        fixed_item=enumeration[where];best=choose(enumeration,'total')
        point_rows=[dict(slot=i,index=k,link=owners[k],point_m=points[0,k].cpu().tolist(),normal=normal[0,k].cpu().tolist(),signed_distance_mm=float(distance[0,k]*1000),absolute_distance_mm=float(distance[0,k].abs()*1000),weighted_distance=float(distance[0,k].abs()*100)) for i,k in enumerate(fixed)]
        detail=dict(row_index=row['row_index'],candidate=row['candidate'],mcp_rad=row['mcp_rad'],pip_rad=row['pip_rad'],geometry_admissible=row['admissible'],
            fixed=fixed_item,reselected=best,nearest_middle_contact=nearest,nearest_middle_link=owners[nearest],nearest_middle_distance_mm=float(distance[0,nearest].abs()*1000),
            fixed_points=point_rows,middle_candidates=[dict(index=k,link=owners[k],point_m=points[0,k].cpu().tolist(),normal=normal[0,k].cpu().tolist()) for k in middle],enumeration=enumeration,
            unknown_containment_directions=row['unknown_containment_directions'])
        all_pose.append(detail);poses[row['row_index']]=pose
        print(f'row {row["row_index"]}: enumerated {len(middle)} middle slots, geometry={row["reason"]}',flush=True)
    save(a.output/'results.json',all_pose)
    for filename,items in [('enumeration.csv',flat),('four_pose_enumeration.csv',[r for r in flat if r['row_index'] in a.diagnostic_rows])]:
        with (a.output/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(items)
    focus=[next(r for r in all_pose if r['row_index']==i) for i in a.diagnostic_rows];save(a.output/'four_pose_details.json',focus)
    point_table=[]
    for detail in focus:
        for point in detail['fixed_points']:
            point_table.append(dict(row_index=detail['row_index'],candidate=detail['candidate'],slot=point['slot'],index=point['index'],link=point['link'],
                signed_distance_mm=point['signed_distance_mm'],absolute_distance_mm=point['absolute_distance_mm'],weighted_distance=point['weighted_distance'],
                **{axis:point['point_m'][i] for i,axis in enumerate(['x_m','y_m','z_m'])},
                **{axis:point['normal'][i] for i,axis in enumerate(['nx','ny','nz'])}))
    with (a.output/'four_pose_points.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(point_table[0]));w.writeheader();w.writerows(point_table)
    delta=[]
    for candidate in sorted({r['candidate'] for r in focus}):
        group=[r for r in focus if r['candidate']==candidate]
        if len(group)==2 and all(r['fixed']['total'] is not None for r in group):
            base=group[0]['fixed'];near=group[1]['fixed']
            delta.append(dict(candidate=candidate,reference_row=group[0]['row_index'],comparison_row=group[1]['row_index'],**{k:near[k]-base[k] for k in ['E_fc_force','E_fc_torque','E_fc','weighted_dis','weighted_pen','weighted_gap','total']},
                middle_distance_change_mm=near['selected_middle_distance_mm']-base['selected_middle_distance_mm']))
    save(a.output/'ranking_attribution.json',delta)
    if not a.diagnostics_only:
        old=json.loads((source/'selected_references.json').read_text())
        for candidate in cfg['candidates']:
            for control,key in [('fixed','fixed'),('reselected','reselected')]:
                pool=[(d,d[key]) for d in all_pose if d['candidate']==candidate and d[key] is not None and d[key]['admissible']]
                if not pool:choices.append(dict(candidate=candidate,control=control,selected=None));continue
                d,item=min(pool,key=lambda x:x[1]['total'])
                if control=='fixed':
                    expected=next(x['selected'] for x in old if x['candidate']==candidate and x['control']=='mesh_gap')
                    if d['row_index']!=expected:raise ValueError('Fixed baseline ranking not reproduced')
                q=dict(raw[candidate]['qpos']);q.update({n:float(poses[d['row_index']][0,9+i]) for i,n in enumerate(names)})
                pending=['semantic_region_not_verified','independent_absolute_geometry_pending','continuous_closure_path_pending','dynamics_pending']
                if d['unknown_containment_directions']:pending.append('open_mesh_containment_unknown')
                choices.append(dict(candidate=candidate,control=control,selected=d['row_index'],middle_contact=item['middle_contact'],middle_link=item['middle_link'],
                    selected_middle_distance_mm=item['selected_middle_distance_mm'],nearest_middle_distance_mm=d['nearest_middle_distance_mm'],total=item['total'],qpos=q,scale=1.,
                    dynamics_admitted=False,pending=pending,geometry_source_row=d['row_index']))
                import plotly.graph_objects as go
                hand.set_parameters(poses[d['row_index']],torch.tensor([fixed[:slot]+[item['middle_contact']]+fixed[slot+1:]],device='cuda'))
                fig=go.Figure(hand.get_plotly_data(0,opacity=.6,with_contact_points=True)+obj.get_plotly_data(0,opacity=.3))
                fig.update_layout(scene_aspectmode='data',title=f'{candidate} {control}; physical contact/region/path unverified')
                fig.write_html(a.output/f'{candidate}_{control}.html',include_plotlyjs=True)
    save(a.output/'selected_references.json',choices)
    save(a.output/'dynamics_admission.json',dict(admitted=False,reason='Static enumeration is not absolute geometry, semantic-region or path certification',expected_direction_trials=36,started=False))
    lines=['# 中指接触槽分项诊断与重选','',f'冻结构型{len(all_pose)}个，组合{len(flat)}组。区域标签默认unknown；不改变原共同可行域，不启动动力学。','', '|候选|近接行−选中行|ΔE_fc|Δ100E_dis|Δ总目标|','|---|---|---:|---:|---:|']
    for d in delta:lines.append(f'|{d["candidate"]}|{d["comparison_row"]}−{d["reference_row"]}|{d["E_fc"]:.6f}|{d["weighted_dis"]:.6f}|{d["total"]:.6f}|')
    lines+=['','|候选|控制|行|中指槽|所选点距离mm|最近候选距离mm|','|---|---|---|---|---:|---:|']
    for c in choices:
        if c['selected'] is not None:lines.append(f'|{c["candidate"]}|{c["control"]}|{c["selected"]}|{c["middle_contact"]}|{c["selected_middle_distance_mm"]:.4f}|{c["nearest_middle_distance_mm"]:.4f}|')
    text='\n'.join(lines)+'\n';(a.output/'REPORT.md').write_text(text)
    links=''.join(f'<p><a href="{c["candidate"]}_{c["control"]}.html">{c["candidate"]} {c["control"]}</a></p>' for c in choices if c['selected'] is not None)
    import plotly.graph_objects as go
    charts=[]
    for d in focus:
        fig=go.Figure(go.Scatter(x=[e['selected_middle_distance_mm'] for e in d['enumeration']],y=[e['total'] for e in d['enumeration']],mode='markers',text=[str(e['middle_contact'])+' '+e['middle_link'] for e in d['enumeration']]))
        fig.update_layout(title=f'row {d["row_index"]}: all 80 slots (unknown semantic regions)',xaxis_title='Selected middle point distance mm',yaxis_title='Target value')
        charts.append(fig.to_html(full_html=False,include_plotlyjs=True if not charts else False))
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><pre>'+html.escape(text)+'</pre>'+links+''.join(charts))
    save(a.output/'status.json',dict(complete=True,poses=len(all_pose),combinations=len(flat),diagnostics_only=a.diagnostics_only,dynamics_started=False))
    print('Complete:',a.output.resolve())


if __name__=='__main__':main()
