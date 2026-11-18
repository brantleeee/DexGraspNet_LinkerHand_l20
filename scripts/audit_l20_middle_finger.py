"""CPU audit: URDF FK vs generation/Plotly, finger seams and object proximity.
No optimization, simulation, or changes to source assets.
"""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,default=22);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    sys.path.insert(0,str(ROOT/'grasp_generation'))
    import torch,transforms3d
    from utils.hand_model import HandModel
    from validate_l20_isaacsim import transform,rotation,axis_rotation
    urdf=ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'
    hand=HandModel(model_path=str(urdf),mesh_path=str(urdf.parent/'meshes'),contact_points_path=str(urdf.parent/'contact_points.json'),penetration_points_path=str(urdf.parent/'penetration_points.json'),n_surface_points=0,device='cpu',force_mesh_sdf=True)
    robot=ET.parse(urdf).getroot();js=robot.findall('joint')
    raw=np.load(ROOT/'data/experiments/l20_clean_drill_20261115/results/scales/scale_1/power-drill-clean-l20.npy',allow_pickle=True)
    obj=trimesh.load(ROOT/'data/meshdata/power-drill-clean-l20/coacd/decomposed.obj',force='mesh',process=False)
    g=raw[a.candidate];fingers=['index','middle','ring','pinky'];result={'candidate':a.candidate,'curl_statistics':{},'sample_count':len(raw),'scale':float(g['scale'])}
    result['source_hashes']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [urdf,ROOT/'grasp_generation/utils/hand_model.py',ROOT/'grasp_generation/utils/energy.py',ROOT/'grasp_generation/utils/optimizer.py',ROOT/'grasp_generation/utils/initializations.py',ROOT/'data/experiments/l20_clean_drill_20261115/results/scales/scale_1/power-drill-clean-l20.npy']}
    for f in fingers:
        values=np.array([r['qpos'][f+'_mcp_pitch'] for r in raw]);result['curl_statistics'][f]=dict(zero_count=int((values<1e-6).sum()),median_rad=float(np.median(values)),max_rad=float(values.max()))
    maxfk=0.;maxplot=0.;seams=[]
    cases=[('zero',{n:0. for n in hand.joints_names}),('equal_curl',{n:(.6 if n.endswith('mcp_pitch') else .8 if n.endswith('_pip') else .8*.8917 if n.endswith('_dip') else 0.) for n in hand.joints_names}),('generated',g['qpos'])]
    world_vertices={}
    for case,q in cases:
        R=transforms3d.euler.euler2mat(*[q.get('WRJR'+k,0.) for k in 'xyz']);t=np.array([q.get('WRJT'+k,0.) for k in 'xyz'])
        pose=t.tolist()+R[:,:2].T.ravel().tolist()+[q[n] for n in hand.joints_names];hand.set_parameters(torch.tensor([pose],dtype=torch.float))
        frames={'base_link':np.eye(4)};pending=js.copy()
        while pending:
            before=len(pending)
            for j in pending[:]:
                parent=j.find('parent').get('link');child=j.find('child').get('link')
                if parent not in frames:continue
                o=j.find('origin');axis=list(map(float,j.find('axis').get('xyz').split()));xyz=list(map(float,o.get('xyz').split()));rpy=list(map(float,o.get('rpy').split()))
                frames[child]=frames[parent]@transform(xyz,rotation(rpy))@transform([0,0,0],axis_rotation(axis,q[j.get('name')]))
                pending.remove(j)
            if before==len(pending):raise ValueError('Bad graph')
        traces={x.name:x for x in hand.get_plotly_data(0)}
        for name,frame in frames.items():
            maxfk=max(maxfk,float(abs(frame-hand.current_status[name].get_matrix()[0].numpy()).max()))
            if name in hand.mesh:
                v=hand.mesh[name]['vertices'].numpy();expected=(v@frame[:3,:3].T+frame[:3,3])@R.T+t
                plot=np.column_stack([traces[name].x,traces[name].y,traces[name].z]);maxplot=max(maxplot,float(abs(plot-expected).max()))
                world_vertices[(case,name)]=plot
        for f in fingers:
            for left,right in [('proximal','middle'),('middle','distal')]:
                x=world_vertices[(case,f+'_'+left)];y=world_vertices[(case,f+'_'+right)]
                seams.append(dict(pose=case,finger=f,pair=left+'-'+right,min_vertex_gap_mm=float(cKDTree(x).query(y)[0].min()*1000)))
    result['max_fk_matrix_difference']=maxfk;result['max_plotly_vertex_difference_m']=maxplot;result['adjacent_mesh_vertex_gaps']=seams
    result['candidate_finger_contact_samples']={}
    for f in fingers:
        info=[]
        for link in ['proximal','middle','distal']:
            v=world_vertices[('generated',f+'_'+link)];v=v[np.linspace(0,len(v)-1,min(256,len(v)),dtype=int)]
            _,dist,_=obj.nearest.on_surface(v)
            info.append(dict(link=link,min_surface_sample_distance_mm=float(dist.min()*1000),within_1mm_count=int((dist<=.001).sum()),samples=len(v)))
        result['candidate_finger_contact_samples'][f]=info
    result['generated_mimic_error_max_rad']=max(abs(float(r['qpos'][f+'_dip'])-.8917*float(r['qpos'][f+'_pip'])) for r in raw for f in fingers)
    # Separate the legacy 20mm keypoint repulsion from real mesh penetration.
    hand.set_parameters(torch.tensor([pose],dtype=torch.float,requires_grad=True))
    penalty=hand.self_penetration()
    grad=torch.autograd.grad(penalty.sum(),hand.hand_pose,retain_graph=True)[0][0,9:]
    result['self_repulsion']={'value':float(penalty.detach()),'saved_value':float(g['E_spen']),
        'mcp_pitch_gradient_unweighted':{f:float(grad[hand.joints_names.index(f+'_mcp_pitch')]) for f in fingers}}
    pts=hand.get_penetraion_keypoints()[0].detach()
    distance=((pts[:,None]-pts[None,:]+1e-13).square().sum(-1)).sqrt()
    distance=torch.where(distance<1e-6,torch.full_like(distance,1e6),distance)
    pair_penalty=(.02-distance).clamp(min=0).numpy()
    owners=[list(hand.mesh)[i] for i in hand.global_index_to_link_index_penetration.tolist()]
    same=np.array([[x==y for y in owners] for x in owners])
    finger=lambda n:n.split('_')[0] if n!='base_link' else 'base'
    same_finger=np.array([[finger(x)==finger(y) for y in owners] for x in owners])
    result['self_repulsion'].update(same_link=float(pair_penalty[same].sum()),same_finger_different_link=float(pair_penalty[same_finger & ~same].sum()),different_finger=float(pair_penalty[~same_finger].sum()),note='Ordered keypoint pairs, raw metres summed, not true mesh penetration depth. Positive gradient at lower MCP bound makes gradient descent request further opening.')
    result['contact_selection']={'total_candidates':hand.n_contact_candidates,'per_link':{n:int(v['contact_candidates'].shape[0]) for n,v in hand.mesh.items()},'optimized_points':4,'per_finger_coverage_enforced':False,'saved_selected_indices_available':'contact_point_indices' in g}
    result['initial_middle_mcp_median_rad']=float(np.median([r['qpos_st']['middle_mcp_pitch'] for r in raw]))
    pts=hand.get_contact_candidates()[0];owners=hand.global_index_to_link_index
    middle_ids=[i for n,i in hand.link_name_to_link_index.items() if n.startswith('middle_')]
    middle_mask=torch.zeros_like(owners,dtype=torch.bool)
    for i in middle_ids:middle_mask |= owners==i
    grad_other=torch.autograd.grad(pts[~middle_mask].square().sum(),hand.hand_pose)[0]
    joint_index=hand.joints_names.index('middle_mcp_pitch')
    jac=torch.autograd.functional.jacobian(lambda x:hand.chain.forward_kinematics(x)['middle_distal'].get_matrix()[0,:3,3],hand.hand_pose[:,9:])[...,joint_index]
    result['kinematic_dependency_check']={'other_branch_position_objective_middle_mcp_gradient':float(grad_other[0,9+joint_index]),'middle_distal_origin_mcp_jacobian_norm_m_per_rad':float(jac.norm()),'note':'Dependency check, not reconstruction of saved contact loss. Original selected contact indices were not saved.'}
    prepared=ROOT/'data/experiments/l20_clean_drill_sdf_20261115_225533/inputs/manifest.json'
    if prepared.exists():
        m=json.loads(prepared.read_text());c=next(c for c in m['candidates'] if c['index']==a.candidate)
        result['sim_input']={'manifest':str(prepared),'urdf_hash_matches':m['urdf_sha256']==hashlib.sha256(urdf.read_bytes()).hexdigest(),'middle_target_correction_rad':{n:float(c['targets'][n]-g['qpos'][n]) for n in hand.joints_names if n.startswith('middle_')},'note':'Prepared input audit only; no actual simulated link poses checked.'}
        arrays=np.load(prepared.parent/'meshes.npz');mesh_errors={}
        for link in m['links']:
            name=link['name']
            if not name.startswith('middle_'):continue
            vertices=[]
            for collider in link['colliders']:
                vertices.append(arrays[collider['key']+'_vertices']@rotation(collider['rpy']).T+collider['xyz'])
            sim_vertices=np.concatenate(vertices);gen_vertices=hand.mesh[name]['vertices'].numpy()
            mesh_errors[name]=max(float(cKDTree(sim_vertices).query(gen_vertices)[0].max()),float(cKDTree(gen_vertices).query(sim_vertices)[0].max()))
        result['sim_input']['middle_local_mesh_vertex_hausdorff_m']=mesh_errors
    result['note']='Vertex seam gap is a sampled geometry diagnostic, not physical hinge separation. Object distances are unsigned sampled proximity, not signed penetration or contact force. No generator source, assets or checkpoints changed.'
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'audit.json').write_text(json.dumps(result,indent=2))
    import plotly.graph_objects as go
    fig=go.Figure();colors=['#ff7f0e','#d62728','#2ca02c','#9467bd']
    for f,color in zip(fingers,colors):
        for link in ['proximal','middle','distal']:
            name=f+'_'+link;v=world_vertices[('generated',name)];faces=hand.mesh[name]['faces'].numpy()
            fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color=color,name=name,showlegend=True))
    v=obj.vertices;faces=obj.faces;fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='gray',opacity=.25,name='object'))
    fig.update_layout(scene_aspectmode='data',title=f'L20 candidate {a.candidate}: middle=red, index=orange, ring=green, pinky=purple; static FK audit')
    fig.write_html(a.output/'finger_comparison.html',include_plotlyjs=True)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
