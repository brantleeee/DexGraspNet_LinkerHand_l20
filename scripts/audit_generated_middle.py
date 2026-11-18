"""Frozen v6-result audit: actual saved contacts, independent gradients and MCP scans."""
import argparse,csv,hashlib,html,json,sys
from pathlib import Path
import numpy as np
import torch
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'grasp_generation'))
from diagnose_l20_middle_objective import pose_from_record
from l20_objective_diagnostics import pair_masks,legacy_pair_loss,gradient


def scalar(v):return float(v.detach().reshape(-1)[0])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'data/experiments/l20_v6_clean_drill_20261117_135743')
    p.add_argument('--candidates',nargs='+',type=int,default=[21,16,7,30])
    p.add_argument('--seed',type=int,default=917);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    if not torch.cuda.is_available():p.error('CUDA is required for frozen objective probes')
    from utils.hand_model import HandModel
    from utils.object_model import ObjectModel
    from utils.energy import cal_energy
    from utils.contact_library import attach_library,normal_alignment,world_contact_normals
    from utils.grasp_results import verify_mesh_identity
    run=a.run.resolve();source=run/'results/scales/scale_1/power-drill-clean-l20.npy';library=run/'contact_library'
    provenance=json.loads((run/'generation_contact_model.json').read_text());data=np.load(source,allow_pickle=True)
    if any(i<0 or i>=len(data) for i in a.candidates):p.error('Invalid candidate IDs')
    for path,h in provenance['code_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise ValueError('Generation code changed: '+path)
    urdf=ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf';mesh=ROOT/'data/meshdata/power-drill-clean-l20/coacd/decomposed.obj'
    verify_mesh_identity(source,mesh);torch.manual_seed(a.seed);np.random.seed(a.seed)
    hand=HandModel(model_path=str(urdf),mesh_path=str(urdf.parent/'meshes'),contact_points_path=str(library/'contact_points.json'),penetration_points_path=str(urdf.parent/'penetration_points.json'),device='cuda',force_mesh_sdf=True)
    identity=attach_library(hand,library)
    if identity['manifest_sha256']!=provenance['manifest_sha256']:raise ValueError('Library does not match generation')
    obj=ObjectModel(str(ROOT/'data/meshdata'),1,2000,device='cuda');obj.initialize(['power-drill-clean-l20'],scales=[1.])
    a.output.mkdir(parents=True);np.save(a.output/'object_surface_points.npy',obj.surface_points_tensor.cpu().numpy())
    owners=[list(hand.mesh)[i] for i in hand.global_index_to_link_index.tolist()]
    key_owners=[list(hand.mesh)[i] for i in hand.global_index_to_link_index_penetration.tolist()]
    tree=ET.parse(urdf).getroot();adjacent=[(j.find('parent').get('link'),j.find('child').get('link')) for j in tree.findall('joint')]
    masks=pair_masks(key_owners,adjacent);names=hand.joints_names
    weights=dict(E_fc=1.,E_dis=100.,E_pen=100.,E_spen=10.,E_joints=1.,E_normal=float(provenance['w_normal']))
    summary=[];order=np.argsort([g['energy'] for g in data]);rank={int(i):k+1 for k,i in enumerate(order)}
    for i,g in enumerate(data):
        contacts=g['contact_point_indices'];q=g['qpos'];selected=[owners[j] for j in contacts]
        assert g['contact_stable_ids']==hand.library_stable_ids[contacts].cpu().tolist()
        row=dict(candidate=i,rank=rank[i],middle_selected=any(x.startswith('middle_') for x in selected),contact_links=selected,
            contact_stable_ids=g['contact_stable_ids'],source_energy=float(g['energy']))
        for finger in ['index','middle','ring','pinky']:
            for suffix in ['mcp_pitch','pip','dip']:row[finger+'_'+suffix]=float(q[finger+'_'+suffix])
        row['max_mimic_error_rad']=max(abs(q[f+'_dip']-.8917*q[f+'_pip']) for f in ['index','middle','ring','pinky'])
        summary.append(row)
    def terms(pose,contacts):
        hand.set_parameters(pose,contacts)
        raw=cal_energy(hand,obj,verbose=True);_,n=obj.cal_distance(hand.contact_points)
        return dict(zip(['E_fc','E_dis','E_pen','E_spen','E_joints'],raw[1:])),normal_alignment(world_contact_normals(hand),n)
    def independent(v,pose):
        g=gradient(v,pose,hand.joint_constraints)
        return {name:float(g[0,9+names.index(name)]) for name in ['middle_mcp_pitch','middle_pip']}
    results=[]
    for i in a.candidates:
        print('candidate',i,'saved contact objective audit',flush=True)
        g=data[i];base,_=pose_from_record(g,hand,'cuda');pose=base.clone().requires_grad_(True)
        contacts=torch.tensor([g['contact_point_indices']],device='cuda',dtype=torch.long)
        values,normal=terms(pose,contacts);values['E_normal']=normal
        rep=legacy_pair_loss(hand.get_penetraion_keypoints()[0])
        categories={k:(rep*torch.tensor(v,device='cuda')).sum().reshape(1) for k,v in masks.items()}
        if not torch.allclose(sum(categories.values()),values['E_spen'],atol=3e-5,rtol=1e-5):raise ValueError('Self-pen decomposition mismatch')
        term_rows={k:dict(value=scalar(v),weight=weights[k],weighted_gradient=independent(weights[k]*v,pose)) for k,v in values.items()}
        pair_rows={k:dict(value=scalar(v),weighted_gradient=independent(10*v,pose)) for k,v in categories.items()}
        total=sum(weights[k]*v for k,v in values.items())
        detail=dict(candidate=i,rank=rank[i],saved_contacts=summary[i]['contact_links'],saved_contact_stable_ids=g['contact_stable_ids'],
            baseline_joints={n:float(g['qpos'][n]) for n in ['middle_mcp_pitch','middle_pip','middle_dip']},terms=term_rows,
            repulsion_categories=pair_rows,total_independent_gradient=independent(total,pose),mcp_scan=[],pair_contributions=[])
        for left in sorted(set(key_owners)):
            for right in sorted(set(key_owners)):
                if left>right or not (left.startswith('middle_') or right.startswith('middle_')):continue
                mask=np.array([x==left for x in key_owners])[:,None]&np.array([x==right for x in key_owners])[None,:]
                if left!=right:mask|=mask.T
                loss=(rep*torch.tensor(mask,device='cuda')).sum()
                if scalar(loss)>.000001:detail['pair_contributions'].append(dict(left=left,right=right,value=scalar(loss),weighted_gradient=independent(10*loss,pose)))
        for delta in [0.,.015,.05,.1,.2,.4]:
            moved=base.clone();idx=9+names.index('middle_mcp_pitch');moved[0,idx]+=delta;moved=hand.joint_constraints.project(moved)
            with torch.no_grad():
                v,n=terms(moved,contacts);v['E_normal']=n;dist,_=obj.cal_distance(hand.get_contact_candidates())
                middle=torch.tensor([s.startswith('middle_') for s in owners],device='cuda')
                selected_middle=[j for j,index in enumerate(g['contact_point_indices']) if owners[index].startswith('middle_')]
                selected_dist,_=obj.cal_distance(hand.contact_points)
            detail['mcp_scan'].append(dict(requested_delta=delta,actual_mcp_rad=float(moved[0,idx]),terms={k:scalar(x) for k,x in v.items()},
                weighted_total=sum(weights[k]*scalar(x) for k,x in v.items()),middle_candidate_min_abs_mm=float(dist[0,middle].abs().min()*1000),
                selected_middle_abs_mm=[float(selected_dist[0,j].abs()*1000) for j in selected_middle]))
        detail['repulsion_finite_difference']={}
        for joint in ['middle_mcp_pitch','middle_pip']:
            idx=9+names.index(joint);endpoints=[]
            for delta in [-.001,.001]:
                moved=base.clone();moved[0,idx]+=delta;moved=hand.joint_constraints.project(moved)
                with torch.no_grad():hand.set_parameters(moved,contacts);loss=hand.self_penetration()
                endpoints.append((float(moved[0,idx]),10*scalar(loss)))
            numeric=(endpoints[1][1]-endpoints[0][1])/(endpoints[1][0]-endpoints[0][0])
            detail['repulsion_finite_difference'][joint]=dict(analytic=term_rows['E_spen']['weighted_gradient'][joint],numeric=numeric,endpoints=endpoints)
        results.append(detail)
    stats={}
    for finger in ['index','middle','ring','pinky']:
        values=np.array([x[finger+'_mcp_pitch'] for x in summary]);stats[finger]=dict(mcp_median=float(np.median(values)),mcp_at_lower=int((values<1e-6).sum()),pip_median=float(np.median([x[finger+'_pip'] for x in summary])))
    status=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generation_contact_provenance=provenance,
        scope='Frozen objective probes with actual saved contacts; no new optimization or dynamics',surface_sampling='NEW fixed 2000 object samples, not saved generation sample; E_pen value/gradient cannot claim exact historical reproduction',
        force_normal_gradient='Original TorchSDF normal detached; new hand-normal FK gradient included; no finite-difference claim for E_fc/E_normal',
        middle_selected_all=sum(x['middle_selected'] for x in summary),middle_selected_top16=sum(summary[int(i)]['middle_selected'] for i in order[:16]),statistics=stats,
        max_mimic_error=max(x['max_mimic_error_rad'] for x in summary))
    (a.output/'results.json').write_text(json.dumps(dict(config=status,all_candidates=summary,probes=results),indent=2)+'\n')
    with (a.output/'candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(sorted(summary,key=lambda x:x['rank']))
    lines=['# v6电钻生成：中指根部不弯曲诊断','','真实保存接触，冻结其他关节；不是新生成或动力学。E_pen采用新固定物体采样，不能当历史梯度重放。','',f'32候选中指被选中{status["middle_selected_all"]}/32，能量前16中{status["middle_selected_top16"]}/16。',
        f'最大mimic误差{status["max_mimic_error"]:.3g}rad。','','|手指|MCP中位rad|MCP下界个数/32|PIP中位rad|','|---|---:|---:|---:|']
    for finger,v in stats.items():lines.append(f'|{finger}|{v["mcp_median"]:.6f}|{v["mcp_at_lower"]}|{v["pip_median"]:.6f}|')
    lines+=['','梯度正值：局部下降方向是减小该角度（MCP进一步张开）；负值：局部下降方向增加该角度。以下不是优化器RMSProp的历史更新。']
    for r in results:
        lines+=['',f'## 候选{r["candidate"]}（能量排名{r["rank"]}）',str(r['saved_contacts']),'','|项|加权MCP梯度|加权PIP梯度|','|---|---:|---:|']
        for k,v in r['terms'].items():lines.append(f'|{k}|{v["weighted_gradient"]["middle_mcp_pitch"]:.6g}|{v["weighted_gradient"]["middle_pip"]:.6g}|')
        lines+=['','自排斥类别：','', '|类别|加权MCP梯度|','|---|---:|']
        for k,v in r['repulsion_categories'].items():lines.append(f'|{k}|{v["weighted_gradient"]["middle_mcp_pitch"]:.6g}|')
        lines+=['','|MCP rad|中指候选最近距离mm|固定接触总目标|E_spen|E_pen（新采样）|','|---|---:|---:|---:|---:|']
        for s in r['mcp_scan']:lines.append(f'|{s["actual_mcp_rad"]:.5f}|{s["middle_candidate_min_abs_mm"]:.5f}|{s["weighted_total"]:.6g}|{s["terms"]["E_spen"]:.6g}|{s["terms"]["E_pen"]:.6g}|')
    lines+=['','**边界**：20mm关键点排斥不等于真实网格自碰撞；本轮不据该项直接认证碰撞，也没有关闭避碰。开放物体符号和有限采样仍有限制。完整逐link对梯度及有限差分见JSON。']
    text='\n'.join(lines)+'\n';(a.output/'REPORT.md').write_text(text)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4));xs=np.arange(32)
    for finger in ['index','middle','ring','pinky']:axes[0].plot(xs,[summary[int(i)][finger+'_mcp_pitch'] for i in order],'.-',label=finger)
    axes[0].set(xlabel='Energy rank - 1',ylabel='MCP pitch (rad)');axes[0].legend()
    for r in results:axes[1].plot([s['actual_mcp_rad'] for s in r['mcp_scan']],[s['middle_candidate_min_abs_mm'] for s in r['mcp_scan']],'.-',label='candidate '+str(r['candidate']))
    axes[1].set(xlabel='Middle MCP pitch (rad)',ylabel='Middle candidate min distance (mm)');axes[1].legend();fig.tight_layout();fig.savefig(a.output/'diagnosis.svg');plt.close(fig)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><img style="max-width:100%" src="diagnosis.svg"><pre>'+html.escape(text)+'</pre>')
    print('Complete:',a.output.resolve(),flush=True)


if __name__=='__main__':main()
