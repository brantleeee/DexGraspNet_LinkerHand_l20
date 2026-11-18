"""Export lowest-energy L20 grasps as individually ranked, standalone HTML files."""
import argparse,hashlib,html,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def rank_candidates(data,k):
    if k<1 or k>len(data):raise ValueError('top-k must be between 1 and number of candidates')
    values=np.array([float(g['energy']) for g in data])
    if not np.isfinite(values).all():raise ValueError('Nonfinite grasp energy')
    order=np.argsort(values,kind='stable').tolist()
    if any('geometry_admitted' in g for g in data):order=[i for i in order if data[i].get('geometry_admitted') is True]
    return order[:k]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-file',type=Path,required=True);p.add_argument('--object-code',required=True)
    p.add_argument('--top-k',type=int,default=16);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.result_file.resolve();output=a.output.resolve()
    if output.exists():p.error('Output must be new')
    data=np.load(source,allow_pickle=True)
    try:order=rank_candidates(data,a.top_k)
    except ValueError as e:p.error(str(e))
    if len({round(float(g['scale']),7) for g in data})!=1:p.error('Rank each exact-scale pose file separately')
    sys.path.insert(0,str(ROOT/'grasp_generation'));sys.path.insert(0,str(ROOT/'scripts'))
    from utils.grasp_results import verify_mesh_identity
    from validate_grasp_dynamics import project_pose
    import xml.etree.ElementTree as ET
    mesh=ROOT/'data/meshdata'/a.object_code/'coacd/decomposed.obj'
    verify_mesh_identity(str(source),str(mesh))
    robot=ET.parse(ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf').getroot()
    for idx in order:
        if not project_pose(robot,data[idx]['qpos'])[1]['source_pose_feasible']:p.error(f'Candidate {idx} violates URDF/mimic')
    import torch,transforms3d,plotly.graph_objects as go
    from utils.hand_model import HandModel
    from utils.object_model import ObjectModel
    hand=HandModel(model_path=str(ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'),mesh_path=str(ROOT/'grasp_generation/mjcf/meshes'),
        contact_points_path=str(ROOT/'grasp_generation/mjcf/contact_points.json'),penetration_points_path=str(ROOT/'grasp_generation/mjcf/penetration_points.json'),n_surface_points=0,device='cpu',force_mesh_sdf=True)
    obj=ObjectModel(data_root_path=str(ROOT/'data/meshdata'),batch_size_each=1,num_samples=0,device='cpu')
    obj.initialize(a.object_code);output.mkdir(parents=True);rows=[]
    for rank,idx in enumerate(order,1):
        g=data[idx];q=g['qpos'];translation=[q['WRJT'+x] for x in 'xyz'];rot=transforms3d.euler.euler2mat(*[q['WRJR'+x] for x in 'xyz'])
        pose=translation+rot[:,:2].T.ravel().tolist()+[q[n] for n in hand.joints_names]
        hand.set_parameters(torch.tensor([pose],dtype=torch.float,device='cpu'))
        obj.object_scale_tensor=torch.tensor([[g['scale']]],dtype=torch.float)
        passed=bool(g['geometry_admitted']) if 'geometry_admitted' in g else bool(g['E_fc']<.3 and g['E_dis']<.005 and g['E_pen']<.001)
        fig=go.Figure(hand.get_plotly_data(i=0,opacity=1,color='lightblue',with_contact_points=False)+obj.get_plotly_data(i=0,color='orange',opacity=1))
        fig.update_layout(scene_aspectmode='data',title=f'{a.object_code} | rank {rank} | candidate {idx} | energy {g["energy"]:.6g} | scale {g["scale"]:.6g} | geometry {"PASS" if passed else "FAIL"}')
        eye=np.asarray(translation,float)
        if np.linalg.norm(eye)>1e-6:
            eye=-1.8*eye/np.linalg.norm(eye)+[0,0,.3];fig.update_layout(scene_camera_eye=dict(zip('xyz',eye)))
        name=f'rank_{rank:02d}_candidate_{idx:03d}.html';fig.write_html(output/name,include_plotlyjs=True)
        rows.append(dict(rank=rank,candidate_index=idx,source_index=int(g.get('source_index',idx)),scale=float(g['scale']),energy=float(g['energy']),geometry_pass=passed,E_fc=float(g['E_fc']),E_dis=float(g['E_dis']),E_pen=float(g['E_pen']),html=name))
    record=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),object_code=a.object_code,ranking='energy ascending; stable source-order ties; no pass-only filtering',rows=rows)
    (output/'ranking.json').write_text(json.dumps(record,indent=2))
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['rank'],r['candidate_index'],f'{r["energy"]:.6g}',r['geometry_pass']])+f'<td><a href="{r["html"]}">打开抓取</a></td></tr>' for r in rows)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>L20抓取能量排名</title><style>body{font:16px system-ui;margin:30px}td,th{border:1px solid #ddd;padding:10px}table{border-collapse:collapse}</style><h1>L20抓取能量排名</h1><p>能量由低到高，未按几何阈值过滤；FAIL仍可进入排名。静态排名不是动力学成功判定。每个HTML独立包含Plotly。</p><table><tr><th>排名</th><th>候选</th><th>能量</th><th>几何阈值</th><th>HTML</th></tr>'+table+'</table>')
    print(f'Exported {len(rows)} ranked grasps: {output}/index.html')

if __name__=='__main__':main()
