"""Paired local joint-grid comparison; not end-to-end generation or dynamics."""
import argparse,json,sys,hashlib,csv,html
from pathlib import Path
import numpy as np
import torch,trimesh
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'grasp_generation'))
from diagnose_l20_middle_objective import pose_from_record,DEFAULT,URDF
from l20_objective_diagnostics import contact_probes
from mesh_gap import truncated_gap,contained


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',nargs='+',type=int,default=[22,18])
    p.add_argument('--mcp-angles',nargs='+',type=float,default=[0,.2,.4,.5,.6,.7,.8,.9,1.,1.1])
    p.add_argument('--pip-deltas',nargs='+',type=float,default=[-.1,0,.1])
    p.add_argument('--margin-mm',type=float,default=1.)
    p.add_argument('--surface-samples',type=int,default=2000)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--probe-index',type=int,default=0)
    p.add_argument('--result-file',type=Path,default=DEFAULT)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new output')
    if not torch.cuda.is_available():p.error('Original E_pen comparison requires CUDA')
    if a.margin_mm<=0 or a.surface_samples<1 or a.probe_index<0 or not np.isfinite(a.mcp_angles+a.pip_deltas+[a.margin_mm]).all():p.error('Invalid numeric arguments')
    if min(a.mcp_angles)<0 or max(a.mcp_angles)>1.4:p.error('MCP outside [0,1.4]')
    from utils.hand_model import HandModel
    from utils.object_model import ObjectModel
    from utils.energy import cal_energy
    from utils.grasp_results import verify_mesh_identity
    meshpath=ROOT/'data/meshdata/power-drill-clean-l20/coacd/decomposed.obj'
    if verify_mesh_identity(a.result_file,meshpath) is None:p.error('Missing mesh identity')
    raw=np.load(a.result_file,allow_pickle=True)
    if any(i<0 or i>=len(raw) for i in a.candidates):p.error('Invalid candidate')
    if any(abs(float(raw[i]['scale'])-1)>1e-7 for i in a.candidates):p.error('This clean-drill experiment requires scale1')
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    hand=HandModel(model_path=str(URDF),mesh_path=str(URDF.parent/'meshes'),contact_points_path=str(URDF.parent/'contact_points.json'),penetration_points_path=str(URDF.parent/'penetration_points.json'),device='cuda')
    obj=ObjectModel(str(ROOT/'data/meshdata'),1,a.surface_samples,device='cuda');obj.initialize(['power-drill-clean-l20'],scales=[1.])
    owners=[list(hand.mesh)[i] for i in hand.global_index_to_link_index.tolist()]
    u,probes=contact_probes(owners,a.probe_index+1,a.seed);ids=probes['grouped4'][a.probe_index]
    contacts=torch.tensor([ids],device='cuda')
    a.output.mkdir(parents=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    files=[Path(__file__),ROOT/'scripts/mesh_gap.py',ROOT/'scripts/check_l20_middle_self_collision.py',URDF,a.result_file,meshpath,ROOT/'grasp_generation/utils/hand_model.py',URDF.parent/'contact_points.json',URDF.parent/'penetration_points.json']
    config.update(sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},contact_ids=ids,contact_links=[owners[i] for i in ids],uniform_draws=u.tolist(),pen_sum_tolerance_m=1e-6,pen_max_tolerance_m=1e-6,note='Same static admissible grid, same contact assignment and object sampling. Ranking comparison, not optimizer convergence or continuous motion. Open-mesh containment unknowns remain.')
    (a.output/'config.json').write_text(json.dumps(config,indent=2));np.save(a.output/'object_surface_points.npy',obj.surface_points_tensor.cpu().numpy())
    results=[];selections=[];links=[]
    import xml.etree.ElementTree as ET
    adjacent={frozenset((j.find('parent').get('link'),j.find('child').get('link'))) for j in ET.parse(URDF).getroot().findall('joint')}
    names=list(hand.mesh);moving=['middle_proximal','middle_middle','middle_distal']
    pairs=sorted({tuple(sorted((x,y))) for x in moving for y in names if x!=y and frozenset((x,y)) not in adjacent})
    config.update(checked_pairs=pairs,excluded_adjacent_pairs=[sorted(x) for x in adjacent],margin_loss='sum(max(0,margin-gap)^2 / margin), metres',weights=dict(distance=100,penetration=100,self_repulsion=10,joints=1))
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    (a.output/'status.json').write_text(json.dumps(dict(complete=False,phase='grid_evaluation')))
    mi=9+hand.joints_names.index('middle_mcp_pitch');pi=9+hand.joints_names.index('middle_pip');margin=a.margin_mm/1000
    for candidate in a.candidates:
        base,_=pose_from_record(raw[candidate],hand,'cuda')
        def energy(pose):
            hand.set_parameters(pose,contacts)
            with torch.no_grad():
                e=cal_energy(hand,obj,verbose=True)
                pen=hand.cal_distance(obj.surface_points_tensor).clamp(min=0)
                d,_=obj.cal_distance(hand.get_contact_candidates())
            middle=torch.tensor([n.startswith('middle_') for n in owners],device='cuda')
            return e,float(pen.max()),float(d[0,middle].abs().min()*1000)
        initial,initial_max,initial_distance=energy(base)
        grid=[(float(base[0,mi]),0.)]+[(m,d) for m in a.mcp_angles for d in a.pip_deltas]
        seen=set();candidate_rows=[];poses={};collision_page_saved=False
        for m,delta in grid:
            pose=base.clone();pose[0,mi]=m;pose[0,pi]+=delta;pose=hand.joint_constraints.project(pose)
            key=(round(float(pose[0,mi]),7),round(float(pose[0,pi]),7))
            if key in seen:continue
            seen.add(key);e,penmax,dist=energy(pose)
            row=dict(candidate=candidate,mcp_rad=float(pose[0,mi]),pip_rad=float(pose[0,pi]),middle_min_distance_mm=dist,E_pen=float(e[3]),max_sampled_pen_m=penmax,legacy_energy=float(e[0]),mesh_gap_energy=None,gap_loss_m=None,admissible=False,reason='',collision_pairs=[],unknown_containment_directions=0)
            if float(e[3])>float(initial[3])+1e-6 or penmax>initial_max+1e-6:
                row['reason']='sampled_hand_object_penetration_increased'
            else:
                meshes={}
                for n,data in hand.mesh.items():
                    frame=hand.current_status[n].get_matrix()[0].cpu().numpy()
                    meshes[n]=trimesh.Trimesh(data['vertices'].cpu().numpy()@frame[:3,:3].T+frame[:3,3],data['faces'].cpu().numpy(),process=False)
                gaps=[];collisions=[];unknown=0
                for x,y in pairs:
                    left,right=meshes[x],meshes[y];gap=truncated_gap(left,right,margin)
                    inside=False
                    if gap>1e-9:
                        for source,target in [(left,right),(right,left)]:
                            verdict=contained(source,target);unknown+=int(verdict is None);inside |= verdict is True
                    if gap<=1e-9 or inside:collisions.append([x,y])
                    gaps.append(dict(left=x,right=y,gap_clipped_m=gap,containment_detected=inside))
                loss=sum((max(0,margin-g['gap_clipped_m'])**2)/margin for g in gaps)
                # Recompose common terms, avoiding cancellation of legacy's large constants.
                cost=float(e[1]+100*e[2]+100*e[3]+e[5])+10*loss
                row.update(mesh_gap_energy=cost,gap_loss_m=loss,admissible=not collisions,reason='self_collision' if collisions else 'passes_sampled_guards',collision_pairs=collisions,unknown_containment_directions=unknown,pair_gaps=gaps)
                if collisions and not collision_page_saved:
                    import plotly.graph_objects as go
                    traces=hand.get_plotly_data(0,opacity=.6)+obj.get_plotly_data(0,opacity=.25)
                    flagged={n for pair in collisions for n in pair}
                    for trace in traces:
                        if trace.name in flagged:trace.color='red'
                    fig=go.Figure(traces);fig.update_layout(scene_aspectmode='data',title=f'{candidate}: rejected self collision {collisions}, MCP={m}, PIPdelta={delta}')
                    filename=f'{candidate}_rejected_collision.html';fig.write_html(a.output/filename,include_plotlyjs=True);links.append(filename);collision_page_saved=True
            row['row_index']=len(results);results.append(row);candidate_rows.append(row);poses[row['row_index']]=pose
            (a.output/'results.json').write_text(json.dumps(results,indent=2))
            print(f'{candidate}: MCP={m:.3f} PIPdelta={delta:.3f} {row["reason"]} distance={dist:.3f}mm',flush=True)
        feasible=[r for r in candidate_rows if r['admissible']]
        for control,key in [('legacy','legacy_energy'),('mesh_gap','mesh_gap_energy')]:
            if not feasible:selections.append(dict(candidate=candidate,control=control,selected=None));continue
            best=min(feasible,key=lambda r:r[key]);pose=poses[best['row_index']];hand.set_parameters(pose,contacts)
            q=dict(raw[candidate]['qpos']);q.update({n:float(pose[0,9+i]) for i,n in enumerate(hand.joints_names)})
            selections.append(dict(candidate=candidate,control=control,selected=best['row_index'],qpos=q,baseline_distance_mm=initial_distance,selected_distance_mm=best['middle_min_distance_mm'],baseline_pen_sum_m=float(initial[3]),selected_pen_sum_m=best['E_pen'],baseline_sampled_pen_max_m=initial_max,selected_sampled_pen_max_m=best['max_sampled_pen_m'],dynamics_validated=False))
            import plotly.graph_objects as go
            fig=go.Figure(hand.get_plotly_data(0,opacity=.6)+obj.get_plotly_data(0,opacity=.35))
            fig.update_layout(scene_aspectmode='data',title=f'{candidate} {control}: static grid selection; not dynamics validated')
            filename=f'{candidate}_{control}.html';fig.write_html(a.output/filename,include_plotlyjs=True);links.append(filename)
    (a.output/'selected_references.json').write_text(json.dumps(selections,indent=2))
    flat=[{k:v for k,v in r.items() if k not in ['pair_gaps','collision_pairs']} for r in results]
    with (a.output/'grid.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# 实际网格间隙排斥静态对照','', '同一局部网格/固定接触点/固定采样；选择结果不是动力学验证。手物穿透护栏沿用原凸包采样代理；开放目标包含仍有未知。间隙截断值=margin代表至少该间隙，非完整距离。','', '|候选|目标|原距离mm|选中距离mm|选中行|','|---|---|---:|---:|---|']
    for r in selections:
        lines.append(f'|{r["candidate"]}|{r["control"]}|{r.get("baseline_distance_mm","—")}|{r.get("selected_distance_mm","—")}|{r["selected"]}|')
    text='\n'.join(lines);(a.output/'REPORT.md').write_text(text)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><pre>'+html.escape(text)+'</pre>'+''.join(f'<p><a href="{n}">{n}</a></p>' for n in links))
    (a.output/'status.json').write_text(json.dumps(dict(complete=True,grid_rows=len(results),dynamics_validated=False)))
    print('Complete:',a.output.resolve())


if __name__=='__main__':main()
