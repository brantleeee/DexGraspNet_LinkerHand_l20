"""Export official generation geometry and kinematics to the native SDF validator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import transforms3d as t3
import trimesh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'external/DexGraspNet_official_compat/grasp_generation'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def safe(name):
    return name.replace('robot0:', '')


def vec(node, key, default='0 0 0'):
    return list(map(float, node.get(key, default).split()))


def rpy(node):
    return list(t3.euler.quat2euler(vec(node, 'quat', '1 0 0 0')))


def prepare(source, meshpath, output, indices=None, top_k=16, density=500., effort=1.):
    import torch
    sys.path.insert(0, str(SOURCE))
    # Official HandModel uses relative mesh paths internally.
    from utils.hand_model import HandModel
    records=np.load(source, allow_pickle=True)
    order=indices if indices is not None else sorted(range(len(records)), key=lambda i: float(records[i]['energy']))[:top_k]
    if not order or len(order)!=len(set(order)) or any(i<0 or i>=len(records) for i in order):
        raise ValueError('Invalid/duplicate candidate indices')
    if indices is None and len(order)!=top_k:raise ValueError('Not enough candidates for top-k')
    scales=[float(records[i]['scale']) for i in order]
    if not np.isfinite(scales).all() or min(scales)<=0 or len(set(scales))!=1:
        raise ValueError('Select one exact positive saved scale; poses are never rescaled')
    if not np.isfinite([density,effort]).all() or min(density,effort)<=0:raise ValueError('Invalid physical parameters')
    if output.exists():raise ValueError('Output must be new')
    xml=SOURCE/'mjcf/shadow_hand_wrist_free.xml'
    tree=ET.parse(xml);body=tree.find('worldbody/body')
    if body.get('name')!='robot0:palm' or body.get('pos','0 0 0')!='0 0 0':raise ValueError('Unexpected root')
    old=Path.cwd()
    try:
        os.chdir(SOURCE)
        hand=HandModel(mjcf_path=str(xml),mesh_path='mjcf/meshes',contact_points_path='mjcf/contact_points.json',penetration_points_path='mjcf/penetration_points.json',n_surface_points=0,device='cpu')
    finally:os.chdir(old)
    arrays={};links=[];joints=[];provenance=[];mapping={}
    bodies={b.get('name'):b for b in tree.findall('.//worldbody//body')}
    offsets={}
    def walk(frame,parent=None,carry=None):
        carry=np.eye(4) if carry is None else carry
        jo=carry@frame.joint.offset.get_matrix()[0].numpy()
        if parent is not None and frame.joint.joint_type=='fixed':
            # PK introduces fixed pre-joint frames; absorb them without extra mass.
            for child in frame.children:walk(child,parent,jo)
            return
        original=frame.link.name
        b=bodies[original.removesuffix('_child')]
        name='base_link' if parent is None else safe(original)
        offset=frame.link.offset.get_matrix()[0].numpy();offsets[original]=offset
        mapping[original]=name
        inert=b.find('inertial')
        if inert is None:raise ValueError('Missing source inertia')
        diag=vec(inert,'diaginertia');colliders=[]
        if original in hand.mesh:
            key='hand_'+name;item=hand.mesh[original]
            arrays[key+'_vertices']=(item['vertices'].numpy()@offset[:3,:3].T+offset[:3,3]).astype(np.float32)
            arrays[key+'_faces']=item['faces'].numpy().astype(np.int32)
            colliders=[dict(key=key,xyz=[0,0,0],rpy=[0,0,0])]
            provenance.append(dict(key=key,path=str(xml),sha256=sha(xml),vertices=len(arrays[key+'_vertices']),faces=len(arrays[key+'_faces']),geometry='official HandModel tessellated collision geometry'))
        com=offset[:3,:3]@np.array(vec(inert,'pos'))+offset[:3,3]
        ir=offset[:3,:3]@t3.euler.euler2mat(*rpy(inert))
        links.append(dict(name=name,mass=float(inert.get('mass')),com=com.tolist(),inertia_rpy=list(t3.euler.mat2euler(ir)),inertia=dict(ixx=diag[0],iyy=diag[1],izz=diag[2],ixy=0.,ixz=0.,iyz=0.),colliders=colliders))
        if parent is not None:
            js=b.findall('joint')
            if len(js)!=1 or vec(js[0],'pos')!=[0,0,0] or js[0].get('ref','0')!='0':raise ValueError('Unsupported joint anchor/reference')
            j=js[0];lo,hi=vec(j,'range')
            joints.append(dict(name=safe(j.get('name')),type='revolute',parent=parent,child=name,xyz=jo[:3,3].tolist(),rpy=list(t3.euler.mat2euler(jo[:3,:3])),axis=frame.joint.axis.numpy().tolist(),limit=dict(lower=lo,upper=hi,effort=effort),mimic=None))
        for child in frame.children:walk(child,name)
    walk(hand.chain._root)
    original_names=hand.chain.get_joint_parameter_names()
    if {safe(n) for n in original_names}!={j['name'] for j in joints}:raise ValueError('Joint set mismatch')
    candidates=[];max_error=0.
    from validate_l20_isaacsim import transform,rotation,axis_rotation
    for rank,i in enumerate(order,1):
        g=records[i];q=g['qpos'];values=np.array([q[n] for n in original_names],dtype=np.float32)
        if not np.isfinite(list(q.values())).all() or not np.isfinite(float(g['energy'])):raise ValueError('Nonfinite source pose/energy')
        targets={safe(n):float(q[n]) for n in original_names}
        for j in joints:
            if not j['limit']['lower']-1e-6<=targets[j['name']]<=j['limit']['upper']+1e-6:raise ValueError(f'Candidate {i}, {j["name"]} outside source joint range: {targets[j["name"]]}; refusing to alter reference')
        # Verify exported FK against the actual generation library at each selected pose.
        fk={'base_link':np.eye(4)}
        for j in joints:fk[j['child']]=fk[j['parent']]@transform(j['xyz'],rotation(j['rpy']))@transform([0,0,0],axis_rotation(j['axis'],targets[j['name']]))
        actual=hand.chain.forward_kinematics(torch.tensor(values).unsqueeze(0))
        for original,name in mapping.items():
            error=float(np.max(np.abs(fk[name]@offsets[original]-actual[original].get_matrix()[0].numpy())))
            max_error=max(max_error,error)
        candidates.append(dict(index=int(i),rank=rank,energy=float(g['energy']),scale=scales[0],qpos={k:float(v) for k,v in q.items()},targets=targets))
    if max_error>2e-6:raise ValueError(f'Exported FK differs from generator: {max_error}')
    obj=trimesh.load(meshpath,force='mesh',process=False);obj.vertices*=scales[0]
    if not obj.is_watertight or not obj.is_winding_consistent or obj.volume<=0:raise ValueError('Object must be a closed oriented solid')
    arrays['object_vertices']=np.asarray(obj.vertices,dtype=np.float32);arrays['object_faces']=np.asarray(obj.faces,dtype=np.int32)
    provenance.append(dict(key='object',path=str(meshpath),sha256=sha(meshpath),applied_scale=scales[0],vertices=len(obj.vertices),faces=len(obj.faces)))
    m=dict(robot='ShadowHand',object='power-drill',object_code='power-drill',scale=scales[0],source=str(source),source_sha256=sha(source),urdf=str(xml),urdf_sha256=sha(xml),links=links,joints=joints,candidates=candidates,meshes=provenance,object_mesh_kind='repaired_nonconvex_solid_union',object_mass_kg=float(obj.volume*density),object_com=obj.center_mass.tolist(),object_inertia=(obj.moment_inertia*density).tolist(),object_extents_m=obj.extents.tolist(),fk_max_matrix_error=max_error,geometry_note='Official generation collision geometry, including simplified boxes/capsules, tessellated by its HandModel. No visual replacement, tendon or mimic invented; all 22 source DOFs independently driven. SDF object is repaired nonconvex solid union.')
    m['generation_hand_model_sha256']=sha(SOURCE/'utils/hand_model.py')
    m['source_mesh_assets']={str(p.relative_to(SOURCE)):sha(p) for p in sorted((SOURCE/'mjcf/meshes').glob('*.obj'))}
    output.mkdir(parents=True);np.savez_compressed(output/'meshes.npz',**arrays)
    (output/'manifest.json').write_text(json.dumps(m,indent=2))
    print(f'Prepared {len(candidates)} ShadowHand poses, {len(joints)} joints; FK error {max_error:.3g}; mass {m["object_mass_kg"]:.6f}kg')
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-file',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--object-mesh',type=Path,default=ROOT/'data/meshdata/power-drill/coacd/decomposed.obj')
    p.add_argument('--indices',nargs='+',type=int);p.add_argument('--top-k',type=int,default=16)
    a=p.parse_args()
    if a.top_k<1:p.error('top-k must be positive')
    prepare(a.result_file.resolve(),a.object_mesh.resolve(),a.output.resolve(),a.indices,a.top_k)

if __name__=='__main__':main()
