"""Local GraspGenX 6-DOF proposals with an explicit frozen ShadowHand closure.

The sweep descriptor is a provisional fingertip-envelope approximation, NOT an
upstream curated ShadowHand asset. No per-grasp finger articulation is predicted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from validate_l20_isaacsim import transform,rotation,axis_rotation
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/lee/robot_learning/open_source_baselines/GraspGenX')
DEFAULT=ROOT/'data/experiments/shadow_drill_valid7_sdf_20261114_235008/inputs'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fk(m,q):
    poses={'base_link':np.eye(4)}
    for j in m['joints']:
        poses[j['child']]=poses[j['parent']]@transform(j['xyz'],rotation(j['rpy']))@transform([0,0,0],axis_rotation(j['axis'],q[j['name']]))
    return poses


def descriptor(inputs,candidate):
    m=json.loads((inputs/'manifest.json').read_text());meshes=np.load(inputs/'meshes.npz')
    c=next(c for c in m['candidates'] if c['index']==candidate)
    close=c['targets'].copy();opened=close.copy()
    for name in close:
        if name[:2] in ('FF','MF','RF','LF') and name[-1] in '012':opened[name]=0.
    half={n:(opened[n]+close[n])/2 for n in close}
    def tips(q):
        poses=fk(m,q);points=[]
        for l in m['links']:
            if 'distal' not in l['name']:continue
            v=meshes[l['colliders'][0]['key']+'_vertices']
            center=v.mean(axis=0);points.append((l['name'],poses[l['name']][:3,:3]@center+poses[l['name']][:3,3]))
        return dict(points)
    points=tips(opened);thumb=next(v for n,v in points.items() if 'thdistal' in n)
    others=np.mean([v for n,v in points.items() if 'thdistal' not in n],axis=0)
    x=others-thumb;x[2]=0.;x/=np.linalg.norm(x);z=np.array([0.,0,1.]);y=np.cross(z,x)
    # palm_from_canonical: +Z along fingers; +X from thumb towards other digits.
    B=np.eye(4);B[:3,:3]=np.column_stack([x,y,z])
    params={}
    for suffix,q in [('open',opened),('mid',half)]:
        pts=np.array(list(tips(q).values()))@B[:3,:3]
        lo=pts.min(axis=0)-.005;hi=pts.max(axis=0)+.005
        params['extents_'+suffix]=(hi-lo).tolist();params['offset_'+suffix]=((hi+lo)/2).tolist()
    params['gripper_type']=2
    result=dict(schema='shadowhand_frozen_closure_v1',closure_source_candidate=candidate,source_inputs=str(inputs),source_manifest_sha256=sha(inputs/'manifest.json'),source_meshes_sha256=sha(inputs/'meshes.npz'),scale=m['scale'],open_targets=opened,close_targets=close,palm_from_canonical=B.tolist(),sweep_params=params,descriptor_status='provisional_heuristic_not_curated',note='Five fingertip-center envelope +5mm padding. Approximate conditioning, not a verified inner swept volume; no named ShadowHand asset exists locally. Finger closure frozen from DexGraspNet candidate; only wrist pose is newly generated.')
    return result,m,meshes


def plot_pose(m,meshes,q,wrist):
    import plotly.graph_objects as go
    poses=fk(m,q);traces=[]
    for l in m['links']:
        for col in l['colliders']:
            key=col['key'];v=meshes[key+'_vertices'];f=meshes[key+'_faces'];T=wrist@poses[l['name']]@transform(col['xyz'],rotation(col['rpy']))
            v=v@T[:3,:3].T+T[:3,3]
            traces.append(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='royalblue',flatshading=True))
    return traces


def prepare(a):
    d,m,meshes=descriptor(a.inputs.resolve(),a.closure_candidate)
    if a.output.exists():raise ValueError('Output must be new')
    a.output.mkdir(parents=True)
    (a.output/'descriptor.json').write_text(json.dumps(d,indent=2))
    import plotly.graph_objects as go
    fig=go.Figure(plot_pose(m,meshes,d['open_targets'],np.eye(4)))
    B=np.array(d['palm_from_canonical'])
    for suffix,color in [('open','orange'),('mid','green')]:
        box=trimesh.creation.box(extents=d['sweep_params']['extents_'+suffix]);box.apply_translation(d['sweep_params']['offset_'+suffix]);box.apply_transform(B)
        v=box.vertices;f=box.faces
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],opacity=.18,color=color,name=suffix))
    fig.update_layout(scene_aspectmode='data',title='ShadowHand provisional fingertip-envelope descriptor (not curated)')
    fig.write_html(a.output/'descriptor.html',include_plotlyjs=True)
    print(a.output/'descriptor.json')


def infer(a):
    d=json.loads(a.descriptor.read_text());inputs=Path(d['source_inputs'])
    if sha(inputs/'manifest.json')!=d['source_manifest_sha256'] or sha(inputs/'meshes.npz')!=d['source_meshes_sha256']:raise ValueError('Frozen hand source changed')
    m=json.loads((inputs/'manifest.json').read_text());meshes=np.load(inputs/'meshes.npz')
    if a.output.exists():raise ValueError('Output must be new')
    a.output.mkdir(parents=True)
    os.environ['GRASPGENX_CHECKPOINT_DIR']=str(REPO/'ext/graspgenx_checkpoints')
    os.environ['GRASPGENX_GRIPPER_CFG_DIR']=str(REPO/'ext/gripper_descriptions')
    sys.path.insert(0,str(REPO))
    import torch
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    meshpath=ROOT/'data/meshdata/power-drill/coacd/decomposed.obj'
    obj=trimesh.load(meshpath,force='mesh',process=False);obj.vertices*=d['scale']
    pc,_=trimesh.sample.sample_surface(obj,3500);mean=pc.mean(axis=0)
    checkpoint=REPO/'ext/graspgenx_checkpoints/release'
    cfg=load_model_cfg(str(checkpoint/'gen'),str(checkpoint/'dis'))
    from graspgenx.x_grippers import make_sweep_volume_gripper_info
    if any(str(cfg[side].gripper_backbone)!='sweep_volume_v2' for side in ['diffusion','discriminator']):raise ValueError('Adapter requires sweep-volume-only checkpoint')
    gripper=make_sweep_volume_gripper_info(**d['sweep_params'],name='shadowhand_frozen_closure')
    sampler=GraspGenXSampler(cfg,gripper_info=gripper)
    grasps,scores=GraspGenXSampler.run_inference(torch.as_tensor(np.asarray(pc-mean,dtype=np.float32),device='cuda'),sampler,num_grasps=a.num_grasps,topk_num_grasps=a.top_k,min_grasps=1,max_tries=1,remove_outliers=False)
    if len(grasps)==0:raise RuntimeError('No proposals returned')
    G=grasps.detach().cpu().numpy();scores=scores.detach().cpu().numpy()
    order=np.argsort(-scores,kind='stable')[:a.top_k];G=G[order];scores=scores[order]
    G[:,:3,3]+=mean;G[:,3,:]=[0,0,0,1]
    B=np.array(d['palm_from_canonical']);wrists=G@np.linalg.inv(B)
    rows=[]
    for rank,(T,score) in enumerate(zip(wrists,scores),1):
        q={'robot0:'+n:float(v) for n,v in d['close_targets'].items()}
        q.update({'WRJT'+n:float(v) for n,v in zip('xyz',T[:3,3])})
        q.update({'WRJR'+n:float(v) for n,v in zip('xyz',Rotation.from_matrix(T[:3,:3]).as_euler('xyz'))})
        rows.append(dict(qpos=q,scale=d['scale'],energy=-float(score),confidence=float(score),generator='GraspGenX_diffusion_with_frozen_closure',energy_semantics='negative_discriminator_score_not_DexGraspNet_energy'))
    np.save(a.output/'power-drill.npy',np.array(rows,dtype=object))
    np.savez_compressed(a.output/'proposals.npz',canonical_poses=G,palm_poses=wrists,scores=scores,point_cloud=pc,point_cloud_center=mean)
    info=dict(descriptor=d,seed=a.seed,requested_samples=a.num_grasps,returned=len(rows),object_mesh_sha256=sha(meshpath),object_extents_m=obj.extents.tolist(),generator_checkpoint=cfg.eval.gen_checkpoint,discriminator_checkpoint=cfg.eval.dis_checkpoint,checkpoint_hashes={p:sha(Path(p)) for p in [cfg.eval.gen_checkpoint,cfg.eval.dis_checkpoint]},physics_validated=False,hand_joint_optimization=False)
    (a.output/'generation.json').write_text(json.dumps(info,indent=2))
    import subprocess
    subprocess.run([str(ROOT/'.venv_l20/bin/python'),str(Path(__file__).resolve()),'render','--output',str(a.output.resolve())],check=True)
    print(f'Generated {len(rows)}: {a.output}/power-drill.npy',flush=True)


def render(a):
    import plotly.graph_objects as go
    info=json.loads((a.output/'generation.json').read_text());d=info['descriptor'];inputs=Path(d['source_inputs'])
    m=json.loads((inputs/'manifest.json').read_text());meshes=np.load(inputs/'meshes.npz')
    raw=np.load(a.output/'proposals.npz');rows=np.load(a.output/'power-drill.npy',allow_pickle=True)
    v=meshes['object_vertices'];f=meshes['object_faces'];scores=raw['scores']
    for rank,T in enumerate(raw['palm_poses'],1):
        traces=plot_pose(m,meshes,d['close_targets'],T)
        traces.append(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='orange'))
        fig=go.Figure(traces);fig.update_layout(scene_aspectmode='data',title=f'GraspGenX rank {rank}, score {scores[rank-1]:.4f}; fixed closure; NOT physics validated')
        fig.write_html(a.output/f'grasp_{rank:02d}.html',include_plotlyjs=True)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>GraspGenX＋固定ShadowHand闭合构型</h1><p>启发式条件描述，未经过物理验证。分数不是稳定成功率。</p>'+''.join(f'<p><a href="grasp_{i:02d}.html">排名{i}，score={float(s):.4f}</a></p>' for i,s in enumerate(scores,1)))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('prepare');s.add_argument('--inputs',type=Path,default=DEFAULT);s.add_argument('--closure-candidate',type=int,default=15);s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('infer');s.add_argument('--descriptor',type=Path,required=True);s.add_argument('--num-grasps',type=int,default=64);s.add_argument('--top-k',type=int,default=16);s.add_argument('--seed',type=int,default=42);s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('render');s.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='infer' and not 1<=a.top_k<=a.num_grasps:p.error('Need 1 <= top-k <= num-grasps')
    {'prepare':prepare,'infer':infer,'render':render}[a.mode](a)

if __name__=='__main__':main()
