"""Native GraspQP ShadowHand synthesis followed by native Isaac Sim SDF evaluation.

Copies assets into this workspace so caches never modify the external checkout.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/lee/robot_learning/open_source_baselines/graspqp')
PYTHON=Path('/home/lee/miniconda3/envs/graspqp/bin/python')
CASES=['gx+','gx-','gy+','gy-','gz+','gz-']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,data):p.write_text(json.dumps(data,indent=2))


def exact_runtime_kinematics():
    # Local environment's PK fork adds sign(theta)*0.01 to every revolute angle.
    # Remove this only in this process, before optimization and export; no disk edits.
    import torch
    from pytorch_kinematics import frame
    def exact(frame_self,theta):
        joint=frame_self.joint;kw=dict(dtype=joint.axis.dtype,device=joint.axis.device)
        if joint.joint_type=='revolute':
            quat=torch.cat([torch.cos(theta/2),torch.sin(theta/2)*joint.axis],dim=-1)
            t=frame.tf.Transform3d(rot=quat,**kw)
        elif joint.joint_type=='prismatic':t=frame.tf.Transform3d(pos=theta*joint.axis,**kw)
        elif joint.joint_type=='fixed':t=frame.tf.Transform3d(default_batch_size=theta.shape[0],**kw)
        else:raise ValueError(joint.joint_type)
        return joint.offset.compose(t)
    frame.Frame.get_transform=exact
    print('Process-local compatibility: exact revolute angles, no hidden 0.01rad offset',flush=True)


def stage_object_mesh(source,scale,folder):
    source=Path(source).resolve()
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid mesh scale')
    obj=trimesh.load(source,force='mesh',process=False)
    if not np.isfinite(obj.vertices).all() or len(obj.faces)==0:raise ValueError('Invalid object mesh')
    obj.vertices*=scale
    folder.mkdir(parents=True)
    if scale==1. and source.suffix.lower()=='.obj':shutil.copy2(source,folder/'remeshed.obj')
    else:obj.export(folder/'remeshed.obj')
    return obj,dict(source_mesh=str(source),mesh_sha256=sha(source),physical_mesh_sha256=sha(folder/'remeshed.obj'),source_scale=scale,object_extents_m=obj.extents.tolist(),object_bounds_m=obj.bounds.tolist(),object_watertight=bool(obj.is_watertight),object_winding_consistent=bool(obj.is_winding_consistent),coordinate_transform='scale only; no centering, axis change, hull or repair',sdf_sign='winding_number' if not obj.is_watertight else 'pseudo_normal',physics_admitted=bool(obj.is_watertight))


def generate(a):
    out=a.output.resolve()
    if out.exists():raise ValueError('Use a new output directory')
    out.mkdir(parents=True)
    assets=out/'hand_assets';assets.mkdir()
    shutil.copytree(REPO/'graspqp/assets/shadow_hand',assets/'shadow_hand')
    mesh=(a.object_mesh or ROOT/'data/meshdata/power-drill/coacd/decomposed.obj').resolve()
    scale=a.mesh_scale if a.mesh_scale is not None else (1. if a.object_mesh else .2554999887943268)
    folder=out/'dataset/power-drill/coacd'
    obj,mesh_info=stage_object_mesh(mesh,scale,folder)
    write(out/'mesh_input.json',mesh_info)
    shutil.copy2(mesh,out/'source_mesh.obj')
    args=['--data_root_path',str(out/'dataset'),'--object_code_list','power-drill','--hand_name','shadow_hand','--energy_type','graspqp','--batch_size',str(a.batch_size),'--n_iter',str(a.n_iter),'--n_contact','12','--no-use_gendexgrasp','--no_plotly','--seed','42']
    write(out/'generation_plan.json',dict(generator='native GraspQP',args=args,mesh_input=mesh_info,mesh_sha256=sha(mesh),physical_mesh_sha256=sha(folder/'remeshed.obj'),compatibility_repairs=['remove_two_fixed_wrist_default_entries','process_local_exact_revolute_angles_without_0.01rad_bias'],source_scale=scale,object_extents_m=obj.extents.tolist(),urdf_sha256=sha(assets/'shadow_hand/shadow_hand.urdf'),hand_dofs=22,note='Native QP force-closure optimization, no GraspGenX descriptor; convex hull used only by native initializer. WARP object SDF uses physical nonconvex mesh.'))
    os.environ['SDF_BACKEND']='WARP'
    os.environ['WANDB_MODE']='disabled'
    exact_runtime_kinematics()
    import graspqp.hands as hands
    hands.ASSET_DIR=str(assets)
    native_factory=hands.get_hand_model
    def fixed_factory(*args,**kwargs):
        model=native_factory(*args,**kwargs)
        if model.n_dofs==22 and model.default_state.numel()==24:
            if list(model.chain.get_joint_parameter_names())[:1]!=['robot0_FFJ3']:raise ValueError('Unexpected native joint order')
            model.default_state=model.default_state[2:]
            print('Compatibility repair: removed two fixed-wrist entries from native default state; 22 optimized joints',flush=True)
        return model
    hands.get_hand_model=fixed_factory
    if not mesh_info['object_watertight']:
        from graspqp.core import ObjectModel
        native_initialize=ObjectModel.initialize
        def winding_initialize(self,*args,**kwargs):
            kwargs['use_winding_number']=True
            return native_initialize(self,*args,**kwargs)
        ObjectModel.initialize=winding_initialize
        print('Open mesh: using WARP generalized winding-number sign; geometry proposals only',flush=True)
    sys.argv=[str(REPO/'scripts/fit.py'),*args]
    runpy.run_path(str(REPO/'scripts/fit.py'),run_name='__main__')


def export(a):
    os.environ['SDF_BACKEND']='WARP'
    import torch
    exact_runtime_kinematics()
    import graspqp.hands as hands
    from validate_l20_isaacsim import transform,rotation,axis_rotation
    out=a.output.resolve();inputs=out/'inputs'
    if inputs.exists():raise ValueError('Inputs exist; refusing overwrite')
    assets=out/'hand_assets';hands.ASSET_DIR=str(assets)
    paths=list((out/'dataset').rglob('power-drill.dexgrasp.pt'))
    if len(paths)!=1:raise ValueError('Final GraspQP prediction not found')
    source=paths[0];data=torch.load(source,map_location='cpu',weights_only=False)
    if data.get('root_frame') not in (None,'gripper_root'):raise ValueError('Non-root grasp frame needs explicit transform')
    hand=hands.get_hand_model('shadow_hand','cpu',n_surface_points=0)
    urdf=assets/'shadow_hand/shadow_hand.urdf';robot=ET.parse(urdf).getroot()
    bodies={l.get('name'):l for l in robot.findall('link')};js=robot.findall('joint')
    root=(set(bodies)-{j.find('child').get('link') for j in js}).pop()
    def alias(n):return 'base_link' if n==root else n
    arrays={};links=[];joints=[];provenance=[];inertia_repairs=[]
    for name,l in bodies.items():
        colliders=[];inert=l.find('inertial');mass=float(inert.find('mass').get('value')) if inert is not None else .0001
        mass=max(mass,.0001)
        if name in hand.mesh:
            v=hand.mesh[name]['vertices'].detach().cpu().numpy();f=hand.mesh[name]['faces'].detach().cpu().numpy();key='hand_'+alias(name)
            arrays[key+'_vertices']=v;arrays[key+'_faces']=f
            colliders=[dict(key=key,xyz=[0,0,0],rpy=[0,0,0])]
            provenance.append(dict(key=key,path=str(urdf),sha256=sha(urdf),vertices=len(v),faces=len(f)))
            extent=np.maximum(np.ptp(v,axis=0),.002)
        else:extent=np.array([.01,.01,.01])
        # Upstream URDF has zero inertia tensors. Record explicit positive box approximation.
        diag=mass/12*np.array([extent[1]**2+extent[2]**2,extent[0]**2+extent[2]**2,extent[0]**2+extent[1]**2])
        origin=inert.find('origin') if inert is not None else None
        com=list(map(float,origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
        links.append(dict(name=alias(name),mass=mass,com=com,inertia_rpy=[0,0,0],inertia=dict(ixx=float(diag[0]),iyy=float(diag[1]),izz=float(diag[2]),ixy=0.,ixz=0.,iyz=0.),colliders=colliders))
        inertia_repairs.append(dict(link=name,mass_kg=mass,diagonal_inertia=diag.tolist(),method='source_mass_and_collision_AABB_box; mass floor 0.1g for massless fixed frames'))
    for j in js:
        origin=j.find('origin');lim=j.find('limit');axis=j.find('axis')
        joints.append(dict(name=j.get('name'),type=j.get('type'),parent=alias(j.find('parent').get('link')),child=alias(j.find('child').get('link')),xyz=list(map(float,origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0],rpy=list(map(float,origin.get('rpy','0 0 0').split())) if origin is not None else [0,0,0],axis=list(map(float,axis.get('xyz').split())) if axis is not None else [1,0,0],limit=dict(lower=float(lim.get('lower')),upper=float(lim.get('upper')),effort=1.) if lim is not None else {},mimic=None))
    ordered=[];pending=joints.copy();seen={'base_link'}
    while pending:
        before=len(pending)
        for j in pending[:]:
            if j['parent'] in seen:ordered.append(j);seen.add(j['child']);pending.remove(j)
        if len(pending)==before:raise ValueError('Disconnected hand graph')
    joints=ordered
    values=data['values'].numpy();order=np.argsort(values)[:a.top_k];candidates=[];rejected=[];maxerr=0.
    for rank,i in enumerate(order,1):
        targets={j['name']:float(data['parameters'][j['name']][i]) for j in joints if j['type']!='fixed'}
        bad=[j['name'] for j in joints if j['type']!='fixed' and not j['limit']['lower']-1e-6<=targets[j['name']]<=j['limit']['upper']+1e-6]
        if bad:rejected.append(dict(index=int(i),rank=rank,reason='joint_range',joints=bad));continue
        fk={'base_link':np.eye(4)};pending=joints.copy()
        while pending:
            before=len(pending)
            for j in pending[:]:
                if j['parent'] not in fk:continue
                fk[j['child']]=fk[j['parent']]@transform(j['xyz'],rotation(j['rpy']))@transform([0,0,0],axis_rotation(j['axis'],targets.get(j['name'],0.)))
                pending.remove(j)
            if len(pending)==before:raise ValueError('Disconnected hand graph')
        actual=hand.chain.forward_kinematics({n:torch.tensor([v]) for n,v in targets.items()})
        for name in bodies:
            if name in actual:
                err=float(abs(fk[alias(name)]-actual[name].get_matrix()[0].numpy()).max())
                maxerr=max(maxerr,err)
        pose=data['parameters']['root_pose'][i].numpy();quat=pose[3:7];rpy=Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz')
        q={**{'WRJT'+n:float(v) for n,v in zip('xyz',pose[:3])},**{'WRJR'+n:float(v) for n,v in zip('xyz',rpy)}}
        candidates.append(dict(index=int(i),rank=rank,energy=float(values[i]),scale=1.,qpos=q,targets=targets))
    if maxerr>2e-6:raise ValueError(f'FK export mismatch: {maxerr}')
    objpath=out/'dataset/power-drill/coacd/remeshed.obj';obj=trimesh.load(objpath,force='mesh',process=False)
    arrays['object_vertices']=np.asarray(obj.vertices,np.float32);arrays['object_faces']=np.asarray(obj.faces,np.int32)
    provenance.append(dict(key='object',path=str(objpath),sha256=sha(objpath),vertices=len(obj.vertices),faces=len(obj.faces)))
    m=dict(robot='GraspQP native ShadowHand',object='power-drill',object_code='power-drill',scale=1.,source=str(source),source_sha256=sha(source),urdf=str(urdf),urdf_sha256=sha(urdf),links=links,joints=joints,candidates=candidates,meshes=provenance,object_mesh_kind='repaired_nonconvex_solid_union_physical_units',object_mass_kg=float(obj.volume*500),object_com=obj.center_mass.tolist(),object_inertia=(obj.moment_inertia*500).tolist(),object_extents_m=obj.extents.tolist(),fk_max_matrix_error=maxerr,inertia_repairs=inertia_repairs,rejected_candidates=rejected,geometry_note='Native GraspQP hand collision geometry; 22 independently optimized DOFs; no DexGraspNet fixed finger template. Hand inertia approximation explicitly recorded because upstream tensors are zero.')
    mesh_info=json.loads((out/'mesh_input.json').read_text()) if (out/'mesh_input.json').exists() else dict(object_watertight=True,physics_admitted=True)
    m['mesh_input']=mesh_info
    if not mesh_info['object_watertight']:
        m['object_mesh_kind']='user_supplied_open_grasp_surface'
        m['mass_estimate_valid']=False
        m['geometry_note']+=' Open source mesh: volume-derived mass/inertia are placeholders; dynamic verification blocked until physical model is specified.'
    inputs.mkdir();np.savez_compressed(inputs/'meshes.npz',**arrays);write(inputs/'manifest.json',m)
    write(out/'input_audit.json',dict(selected=len(order),admitted=len(candidates),rejected=rejected,fk_error=maxerr,hand_inertias_approximate=True))
    subprocess.run([str(ROOT/'.venv_l20/bin/python'),str(Path(__file__).resolve()),'render','--output',str(out)],cwd=ROOT,check=True)
    print('Exported',len(candidates),'valid candidates;',len(rejected),'range rejected; FK',maxerr,flush=True)


def render(a):
    import plotly.graph_objects as go
    from graspgenx_shadow_drill import plot_pose
    out=a.output.resolve();m=json.loads((out/'inputs/manifest.json').read_text());meshes=np.load(out/'inputs/meshes.npz')
    ordered=[];pending=m['joints'].copy();seen={'base_link'}
    while pending:
        before=len(pending)
        for j in pending[:]:
            if j['parent'] in seen:ordered.append(j);seen.add(j['child']);pending.remove(j)
        if len(pending)==before:raise ValueError('Disconnected hand graph')
    m['joints']=ordered
    v=meshes['object_vertices'];f=meshes['object_faces']
    from validate_l20_isaacsim import transform,rotation
    for c in m['candidates']:
        q=c['qpos'];T=transform([q['WRJT'+k] for k in 'xyz'],rotation([q['WRJR'+k] for k in 'xyz']))
        traces=plot_pose(m,meshes,{j['name']:c['targets'].get(j['name'],0.) for j in m['joints']},T)
        traces.append(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='orange'))
        fig=go.Figure(traces);fig.update_layout(scene_aspectmode='data',title=f'Native GraspQP ShadowHand, candidate {c["index"]}, energy {c["energy"]:.4g}, not physics validated')
        fig.write_html(out/f'grasp_{c["index"]:03d}.html',include_plotlyjs=True)
    (out/'generation_index.html').write_text('<!doctype html><meta charset="utf-8"><h1>GraspQP 原生ShadowHand全关节生成</h1>'+''.join(f'<p><a href="grasp_{c["index"]:03d}.html">排名{c["rank"]}, 原下标{c["index"]}</a></p>' for c in m['candidates']))


def validate(a):
    from replay_l20_isaacsim_html import export as replay
    out=a.output.resolve();inputs=out/'inputs';m=json.loads((inputs/'manifest.json').read_text());rows=[]
    if not m.get('mesh_input',{}).get('physics_admitted',True):raise ValueError('Open grasp-surface mesh is not admitted for dynamics: provide a closed physical mesh/inertia model first')
    def report():
        write(out/'results.json',rows)
        count=sum(r['completed'] for r in rows);passed=sum(r['completed'] and r['result']['passed'] for r in rows)
        (out/'REPORT.md').write_text(f'# GraspQP ShadowHand 电钻动力学验证\n\n已尝试{len(rows)}/{len(m["candidates"])*len(CASES)}方向，完成{count}、严格通过{passed}、执行异常{len(rows)-count}。\n\n逐方向与轨迹回放见index.html及results.json。关节越界拒绝见input_audit.json。物理模型包含显式记录的手部惯量近似；不能等同实测硬件。\n')
        write(out/'status.json',dict(completed=len(rows),expected=len(m['candidates'])*len(CASES),all_processes_complete=bool(rows and len(rows)==len(m['candidates'])*len(CASES) and all(r['completed'] for r in rows))))
        table=[]
        for c in m['candidates']:
            rr={r['case']:r for r in rows if r['candidate']==c['index']};cells=[]
            for case in CASES:
                r=rr.get(case);label='待运行' if r is None else '执行异常' if not r['completed'] else '通过' if r['result']['passed'] else '失败'
                cells.append(f'<td><a href="candidate_{c["index"]:03d}/{case}/replay.html">{label}</a></td>')
            table.append(f'<tr><td>{c["rank"]}</td><td><a href="grasp_{c["index"]:03d}.html">{c["index"]}</a></td>'+''.join(cells)+'</tr>')
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>GraspQP 原生ShadowHand 电钻 SDF验证</h1><p>22关节独立优化；原生Isaac Sim direct10秒/方向；手部原URDF零惯量以包围盒近似补齐，非实测硬件标定。候选拒绝记录见input_audit.json。蓝色手部/橙色物体回放。</p><table border="1"><tr><th>排名</th><th>下标</th>'+''.join('<th>'+case+'</th>' for case in CASES)+'</tr>'+''.join(table)+'</table>')
    report()
    for c in m['candidates']:
        for case in CASES:
            folder=out/f'candidate_{c["index"]:03d}'/case;folder.parent.mkdir(exist_ok=True,parents=True)
            if folder.exists():raise ValueError('Trial exists; use new output')
            cmd=['bash',str(ROOT/'scripts/run_l20_isaacsim.sh'),'--inputs',str(inputs),'--output',str(folder),'--indices',str(c['index']),'--cases',case,'--protocol','direct','--steps','2400','--sdf-resolution','256']
            print(f'Native GraspQP ShadowHand {c["index"]} {case}',flush=True)
            with folder.with_suffix('.log').open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            record=dict(candidate=c['index'],case=case,completed=False,returncode=r.returncode)
            if r.returncode==0 and (folder/'summary.json').exists():
                result=json.loads((folder/'summary.json').read_text())['results'][0];record.update(completed=True,result=result)
                replay(folder,inputs,folder/'replay.html')
            rows.append(record);report()
    if not rows or not all(r['completed'] for r in rows):raise SystemExit('Validation has execution errors; retain outputs and inspect logs')


def pipeline(a):
    cmd=[str(PYTHON),str(Path(__file__).resolve()),'generate','--output',str(a.output.resolve()),'--batch-size',str(a.batch_size),'--n-iter',str(a.n_iter)]
    if a.object_mesh:cmd+=['--object-mesh',str(a.object_mesh.resolve())]
    if a.mesh_scale is not None:cmd+=['--mesh-scale',str(a.mesh_scale)]
    subprocess.run(cmd,cwd=ROOT,check=True)
    subprocess.run([str(PYTHON),str(Path(__file__).resolve()),'export','--output',str(a.output.resolve()),'--top-k',str(a.top_k)],cwd=ROOT,check=True)
    if a.mode=='synthesize':return
    subprocess.run([str(ROOT/'.venv_l20/bin/python'),str(Path(__file__).resolve()),'validate','--output',str(a.output.resolve())],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['generate','export','render','validate','pipeline','synthesize']);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--n-iter',type=int,default=3000);p.add_argument('--top-k',type=int,default=16);p.add_argument('--object-mesh',type=Path);p.add_argument('--mesh-scale',type=float);a=p.parse_args()
    if min(a.batch_size,a.n_iter,a.top_k)<1:p.error('Positive counts required')
    (pipeline if a.mode=='synthesize' else globals()[a.mode])(a)
if __name__=='__main__':main()
