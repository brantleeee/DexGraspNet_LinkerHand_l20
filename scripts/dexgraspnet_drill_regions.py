"""Region-constrained DexGraspNet ShadowHand synthesis, no dynamics.

Full object geometry for collision; SAFE_GRASP only for contact attraction.
Semantic clearance is sampled/soft and never a certified whole-hand clearance.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1]
OFFICIAL=ROOT/'external/DexGraspNet_official_compat/grasp_generation'
DEFAULT=Path('/home/lee/robot_learning/pose_estimation/rm_arm_mujoco/assets/objects/power_drill/grasp_clean/region_exports')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read_regions(folder):
    manifest=json.loads((folder/'manifest.json').read_text());source=Path(manifest['source_obj'])
    if sha(source)!=manifest['source_sha256']:raise ValueError('Region source mesh changed')
    a=np.load(folder/'semantic_surface.npz');v=a['vertices'];f=a['faces'];labels=a['face_labels']
    obj=trimesh.load(source,force='mesh',process=False)
    if a['obj_sha256'].item()!=sha(source) or not np.array_equal(f,obj.faces) or not np.allclose(v,obj.vertices,rtol=0,atol=1e-12):raise ValueError('Label/geometry identity mismatch')
    if len(labels)!=len(f) or not np.isin(labels,[0,1,2,3,4]).all() or not (labels==0).any():raise ValueError('Invalid/no SAFE_GRASP labels')
    for name,spec in manifest['classes'].items():
        ids=np.load(folder/spec['face_ids']);expected=np.flatnonzero(labels==spec['class_id'])
        if not np.array_equal(np.sort(ids),expected):raise ValueError(f'Region face map mismatch: {name}')
    return manifest,obj,v,f,labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--regions',type=Path,default=DEFAULT);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--n-iter',type=int,default=6000);p.add_argument('--top-k',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--clearance',type=float,default=.002);p.add_argument('--w-semantic',type=float,default=100.)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():p.error('Output must be new')
    if not 1<=a.top_k<=a.batch_size or a.n_iter<1 or not np.isfinite([a.clearance,a.w_semantic]).all() or min(a.clearance,a.w_semantic)<0:p.error('Invalid parameters')
    manifest,surface,v,f,labels=read_regions(a.regions)
    out.mkdir(parents=True);frozen=out/'region_source';shutil.copytree(a.regions,frozen)
    shutil.copy2(manifest['source_obj'],out/'generation_surface.obj')
    # Build a closed full-object distance model from the same frozen surface.
    from validate_clean_graspqp_drill import closed_proxy
    solid,repair=closed_proxy(surface);meshdir=out/'meshdata/power-drill/coacd';meshdir.mkdir(parents=True)
    solid.export(meshdir/'decomposed.obj')
    config=dict(method='DexGraspNet regional extension',hand='official compatibility ShadowHand',seed=a.seed,batch_size=a.batch_size,n_iter=a.n_iter,top_k=a.top_k,object_scale=1.,bounds_m=surface.bounds.tolist(),extents_m=surface.extents.tolist(),source_sha256=sha(out/'generation_surface.obj'),region_manifest_sha256=sha(a.regions/'manifest.json'),semantic_surface_sha256=sha(a.regions/'semantic_surface.npz'),label_counts={str(i):int((labels==i).sum()) for i in range(5)},contact_classes=[0],excluded_contact_classes=[1,2,3,4],full_collision_mesh_sha256=sha(meshdir/'decomposed.obj'),repair=repair,clearance_m=a.clearance,w_semantic=a.w_semantic,surface_samples_for_clearance=128,hard_joint_projection=True,proposal_graph_detached_each_iteration=True,initialization='SAFE_GRASP area samples + full solid outward normals, wrist stand-off 30-70mm; not paper 200mm hull expansion',limitation='Sampled soft semantic clearance, not a continuous whole-hand or swept-path certificate; no physics validation')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    import torch
    import transforms3d
    sys.path.insert(0,str(OFFICIAL))
    from utils.hand_model import HandModel
    from utils.object_model import ObjectModel
    from utils.energy import cal_energy
    from utils.optimizer import Annealing
    from torchsdf import compute_sdf,index_vertices_by_faces
    torch.manual_seed(a.seed);np.random.seed(a.seed);device='cuda'
    old=Path.cwd()
    try:
        os.chdir(OFFICIAL)
        hand=HandModel('mjcf/shadow_hand_wrist_free.xml','mjcf/meshes','mjcf/contact_points.json','mjcf/penetration_points.json',n_surface_points=128,device=device)
    finally:os.chdir(old)
    obj=ObjectModel(str(out/'meshdata'),a.batch_size,num_samples=2000,device=device);obj.scale_choice=torch.tensor([1.],device=device);obj.initialize('power-drill')
    verts=torch.tensor(v,dtype=torch.float,device=device)
    safe_faces=index_vertices_by_faces(verts,torch.tensor(f[labels==0],device=device,dtype=torch.long))
    excluded_faces=index_vertices_by_faces(verts,torch.tensor(f[labels!=0],device=device,dtype=torch.long))
    safe=trimesh.Trimesh(v,f[labels==0],process=False);points,_=trimesh.sample.sample_surface(safe,a.batch_size)
    _,_,nearest=solid.nearest.on_surface(points);normal=solid.face_normals[nearest];n=-normal
    rotations=[];translations=[]
    for point,direction in zip(points,n):
        rg=transforms3d.euler.euler2mat(math.atan2(direction[1],direction[0])-math.pi/2,-math.acos(np.clip(direction[2],-1,1)),0,axes='rzxz')
        rl=transforms3d.euler.euler2mat(np.random.uniform(0,2*math.pi),np.random.uniform(-math.pi/6,math.pi/6),np.random.uniform(0,2*math.pi),axes='rzxz')
        axis=rg@rl@np.array([0.,0,1.]);translations.append(point-np.random.uniform(.03,.07)*axis)
        rotations.append(rg@rl@transforms3d.euler.euler2mat(0,-math.pi/3,0,axes='rzxz'))
    mu=torch.tensor([.1,0,.6,0,0,0,.6,0,-.1,0,.6,0,0,-.2,0,.6,0,0,1.2,0,-.2,0],device=device)
    q=mu+.1*(hand.joints_upper-hand.joints_lower)*torch.randn(a.batch_size,22,device=device);q=torch.maximum(torch.minimum(q,hand.joints_upper),hand.joints_lower)
    pose=torch.cat([torch.tensor(np.array(translations),device=device,dtype=torch.float),torch.tensor(np.array(rotations),device=device,dtype=torch.float).transpose(1,2)[:,:2].reshape(-1,6),q],dim=1).requires_grad_()
    hand.set_parameters(pose,torch.randint(hand.n_contact_candidates,(a.batch_size,4),device=device));initial=pose.detach().clone()
    def unsigned(points,faces):
        shape=points.shape[:-1];squared,*_=compute_sdf(points.reshape(-1,3).contiguous(),faces)
        return torch.sqrt(squared+1e-12).reshape(shape)
    def energy():
        _,fc,old_dis,pen,spen,joints=cal_energy(hand,obj,verbose=True)
        dis=unsigned(hand.contact_points,safe_faces).sum(-1)
        dist=unsigned(hand.get_surface_points(),excluded_faces)
        sem=torch.relu(a.clearance-dist).sum(-1)
        values=torch.stack([fc,dis,pen,spen,joints,sem],dim=1)
        return fc+100*dis+100*pen+10*spen+joints+a.w_semantic*sem,values
    opt=Annealing(hand,device=device);e,values=energy();e.sum().backward(retain_graph=True)
    for step in range(1,a.n_iter+1):
        opt.try_step()
        # This is an explicitly documented projected variant; never clips an exported pose later.
        projected=torch.cat([hand.hand_pose[:,:9],torch.maximum(torch.minimum(hand.hand_pose[:,9:],hand.joints_upper),hand.joints_lower)],dim=1)
        hand.set_parameters(projected.detach().requires_grad_(),hand.contact_point_indices)
        opt.zero_grad();new_e,new_values=energy()
        if not torch.isfinite(new_e).all():raise RuntimeError(f'Nonfinite objective at step {step}')
        new_e.sum().backward(retain_graph=True)
        with torch.no_grad():
            accepted,_=opt.accept_step(e,new_e);e[accepted]=new_e[accepted];values[accepted]=new_values[accepted]
        if step==1 or step%100==0 or step==a.n_iter:print(f'iteration {step}/{a.n_iter}, median energy={e.median().item():.6g}, median safe_distance_sum={values[:,1].median().item():.6g}, median semantic_penalty={values[:,5].median().item():.6g}',flush=True)
    # Recompute metrics from final accepted state rather than stale rejected proposals.
    with torch.no_grad():e,values=energy();pose=hand.hand_pose.detach().cpu().numpy();contacts=hand.contact_points.detach().cpu().numpy();surf=hand.get_surface_points().detach().cpu().numpy();safe_dist=unsigned(hand.contact_points,safe_faces).detach().cpu().numpy();ex_dist=unsigned(hand.get_surface_points(),excluded_faces).detach().cpu().numpy()
    nearest_labels=[]
    for pts in contacts:
        _,_,idx=surface.nearest.on_surface(pts);nearest_labels.append(labels[idx].tolist())
    rows=[]
    from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
    R=robust_compute_rotation_matrix_from_ortho6d(torch.tensor(pose[:,3:9])).numpy()
    for i in range(a.batch_size):
        qpos={n:float(x) for n,x in zip(hand.joints_names,pose[i,9:])};qpos.update({'WRJT'+n:float(x) for n,x in zip('xyz',pose[i,:3])});qpos.update({'WRJR'+n:float(x) for n,x in zip('xyz',transforms3d.euler.mat2euler(R[i]))})
        metrics=dict(zip(['E_fc','E_dis','E_pen','E_spen','E_joints','E_semantic'],map(float,values[i].detach().cpu())))
        rows.append(dict(scale=1.,qpos=qpos,energy=float(e[i]),**metrics,nearest_contact_class_ids=nearest_labels[i],max_safe_contact_distance_m=float(safe_dist[i].max()),min_excluded_surface_sample_distance_m=float(ex_dist[i].min()),semantic_verified=False,physics_validated=False))
    (out/'results').mkdir();np.save(out/'results/power-drill.npy',np.array(rows,dtype=object));np.savez_compressed(out/'states.npz',hand_pose=pose,initial_hand_pose=initial.cpu().numpy(),contact_points=contacts,hand_surface_points=surf,nearest_contact_labels=nearest_labels,safe_contact_distances=safe_dist,excluded_sample_distances=ex_dist)
    order=np.argsort([r['energy'] for r in rows],kind='stable')[:a.top_k]
    # Separate geometry rendering in this existing environment; no browser server.
    import plotly.graph_objects as go
    colors=np.array(['#00d246','#2878ff','#ffdc00','#eb2323','#969696'])[labels]
    links=[]
    for rank,i in enumerate(order,1):
        hand.set_parameters(torch.tensor(pose[i:i+1],device=device))
        fig=go.Figure(hand.get_plotly_data(0,color='lightblue',opacity=1))
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],facecolor=colors,opacity=1))
        name=f'grasp_{rank:02d}_candidate_{i:03d}.html';fig.update_layout(scene_aspectmode='data',title=f'Regional DexGraspNet ShadowHand rank {rank}, candidate {i}; semantic/physics NOT certified');fig.write_html(out/name,include_plotlyjs=True)
        links.append(f'<p><a href="{name}">排名{rank} 原下标{i}，能量{rows[i]["energy"]:.5g}，接触类别{nearest_labels[i]}</a></p>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>区域约束DexGraspNet＋ShadowHand</h1><p>绿色SAFE为接触目标；其他区域软避让；完整电钻参与穿透优化。采样约束不能证明全手间隙或动态稳定。</p>'+''.join(links))
    summary=dict(generated=len(rows),exported_html=len(order),all_selected_contacts_nearest_safe_count=sum(all(x==0 for x in r['nearest_contact_class_ids']) for r in rows),contact_distance_below_1mm_count=sum(r['max_safe_contact_distance_m']<=.001 for r in rows),semantic_verified=False,physics_validated=False,ranking_indices=order.tolist())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
