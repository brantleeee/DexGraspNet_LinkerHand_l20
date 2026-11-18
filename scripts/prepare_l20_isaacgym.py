"""Prepare the existing five-object L20 pose batches for adapted official Gym screening."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from l20_object_catalog import OBJECTS,ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adjust(poses,names,code,scale,device):
    # Original normal-offset gradient construction, adapted to L20 independent joints.
    import torch
    sys.path.insert(0,str(ROOT/'grasp_generation'))
    from utils.hand_model import HandModel
    from utils.object_model import ObjectModel
    torch.manual_seed(0)
    hand=HandModel(model_path=str(ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'),
        mesh_path=str(ROOT/'grasp_generation/mjcf/meshes'),
        contact_points_path=str(ROOT/'grasp_generation/mjcf/contact_points.json'),
        penetration_points_path=str(ROOT/'grasp_generation/mjcf/penetration_points.json'),
        n_surface_points=2000,device=device)
    if list(hand.joints_names)!=names:raise ValueError('Preparation joint order differs from HandModel')
    state=torch.tensor(poses,dtype=torch.float32,device=device,requires_grad=True)
    hand.set_parameters(state)
    model=ObjectModel(str(ROOT/'data/meshdata'),len(poses),num_samples=0,device=device)
    model.initialize(code,scales=[scale])
    points=[];normals=[]
    for link in hand.mesh:
        sample=hand.mesh[link]['surface_points']
        if len(sample)==0:continue
        surface=hand.current_status[link].transform_points(sample).expand(len(poses),-1,3)
        surface=surface@hand.global_rotation.transpose(1,2)+hand.global_translation.unsqueeze(1)
        distance,normal=model.cal_distance(surface)
        nearest=distance.argmax(dim=1)
        selected=nearest[:,None,None].expand(-1,1,3)
        admitted=(-distance.gather(1,nearest[:,None])<.001)[:,:,None]
        points.append(torch.where(admitted,surface.gather(1,selected),torch.zeros_like(surface[:,:1])))
        normals.append(torch.where(admitted,normal.gather(1,selected),torch.zeros_like(surface[:,:1])))
    points=torch.cat(points,dim=1);normals=torch.cat(normals,dim=1)
    ((points+normals*.001).detach()-points).square().sum().backward()
    gradient=hand.joint_constraints.pullback(state.grad)
    candidate=state.detach().clone();candidate[:,9:]+=gradient[:,9:]*500
    projected=hand.joint_constraints.project(candidate)
    if not torch.isfinite(projected).all():raise ValueError('Nonfinite adjusted targets')
    return projected[:,9:].detach().cpu().numpy(),dict(
        max_projection_rad=float((projected[:,9:]-candidate[:,9:]).abs().max().item()),
        max_adjustment_rad=float((projected[:,9:]-state[:,9:]).abs().max().item()))


def export_object(mesh_path,folder):
    folder.mkdir(parents=True)
    pieces=list(trimesh.load(mesh_path,force='mesh',process=False).split(only_watertight=False))
    if not pieces:raise ValueError('Empty object mesh')
    robot=ET.Element('robot',name='object')
    for i,piece in enumerate(pieces):
        name=f'piece_{i:03d}';piece.export(folder/f'{name}.obj')
        link=ET.SubElement(robot,'link',name=name)
        for kind in ['visual','collision']:
            node=ET.SubElement(link,kind);geom=ET.SubElement(node,'geometry')
            ET.SubElement(geom,'mesh',filename=f'{name}.obj',scale='1 1 1')
        if i:
            joint=ET.SubElement(robot,'joint',name=f'fixed_{i}',type='fixed')
            ET.SubElement(joint,'parent',link='piece_000');ET.SubElement(joint,'child',link=name)
    ET.ElementTree(robot).write(folder/'object.urdf')
    return len(pieces)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'data/experiments/l20_five_objects_acquire_10s_20261105')
    p.add_argument('--objects',nargs='+',choices=OBJECTS,default=list(OBJECTS))
    p.add_argument('--variants',nargs='+',choices=['official','no-force'],default=['official','no-force'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    if len(set(a.objects))!=len(a.objects) or len(set(a.variants))!=len(a.variants):p.error('Duplicate selections')
    manifests={obj:json.loads((a.results/obj/'inputs/manifest.json').read_text()) for obj in a.objects}
    # Check exact original data before writing any output.
    records={}
    for obj,m in manifests.items():
        if m['object']!=obj or digest(m['source'])!=m['source_sha256']:raise ValueError(f'Source changed: {obj}')
        if digest(m['urdf'])!=m['urdf_sha256']:raise ValueError('URDF changed')
        for mesh in m['meshes']:
            if digest(mesh['path'])!=mesh['sha256']:raise ValueError(f'Mesh changed: {mesh["path"]}')
        data=np.load(m['source'],allow_pickle=True)
        for c in m['candidates']:
            r=data[c['index']]
            if not c['feasibility']['source_pose_feasible']:raise ValueError('Refusing silently projected original pose')
            if abs(float(r['scale'])-c['scale'])>1e-7 or r['qpos']!=c['qpos']:raise ValueError('Saved candidate mismatch')
            if not np.isfinite(float(r['E_pen'])):raise ValueError('Invalid saved E_pen')
        records[obj]=data
    a.output.mkdir(parents=True)
    robot=ET.parse(ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf')
    joints=[j for j in robot.getroot().findall('joint') if j.get('type')!='fixed']
    # Match the FK order used by pytorch_kinematics rather than relying on XML order.
    import pytorch_kinematics as pk
    chain=pk.build_chain_from_urdf(ET.tostring(robot.getroot()).decode())
    names=list(chain.get_joint_parameter_names())
    mimics=[dict(child=j.get('name'),**j.find('mimic').attrib) for j in joints if j.find('mimic') is not None]
    hand=a.output/'assets/hand';hand.mkdir(parents=True)
    for node in robot.getroot().iter('mesh'):
        source=(ROOT/'grasp_generation/mjcf'/node.get('filename')).resolve()
        filename=source.stem+'.obj'
        trimesh.load(source,force='mesh').export(hand/filename)
        node.set('filename',filename)
    for j in joints:
        mimic=j.find('mimic')
        if mimic is not None:j.remove(mimic)
    robot.write(hand/'l20.urdf')
    manifest=dict(schema=1,hand='L20',joint_names=names,mimics=mimics,variants=a.variants,
        mimic_mode='consistent_position_targets_no_hard_coupling',
        contact_adjustment=dict(thres_cont=.001,dis_move=.001,grad_move=500,
            adaptation='L20 sampled links, gradient pullback through mimic, hard URDF projection; saved E_pen not recomputed'),
        collision='Gym convex hull per hand mesh and per existing object connected component; not SDF',objects={})
    import transforms3d
    for obj,m in manifests.items():
        rows=m['candidates'];data=records[obj];poses=[];quats=[]
        for c in rows:
            q=c['qpos'];euler=[q['WRJR'+v] for v in 'xyz']
            r=transforms3d.euler.euler2mat(*euler)
            poses.append([q['WRJT'+v] for v in 'xyz']+r[:,:2].T.ravel().tolist()+[q[n] for n in names])
            quats.append(transforms3d.euler.euler2quat(*euler))
        poses=np.asarray(poses);raw=poses[:,9:];arrays=dict(raw_qpos=raw,translations=poses[:,:3],rotations=quats,
            scales=[c['scale'] for c in rows],indices=[c['index'] for c in rows],E_pen=[data[c['index']]['E_pen'] for c in rows])
        info={}
        if 'official' in a.variants:arrays['force_qpos'],info=adjust(poses,names,m['object_code'],m['scale'],a.device)
        mesh=next(v for v in m['meshes'] if v['key']=='object')
        pieces=export_object(Path(mesh['path']),a.output/'assets'/obj)
        np.savez(a.output/f'{obj}.npz',**arrays)
        manifest['objects'][obj]=dict(count=len(rows),source_manifest=m,contact_adjustment_audit=info,object_components=pieces)
        print(f'{obj}: prepared {len(rows)} candidates; object components={pieces}',flush=True)
    manifest['files_sha256']={str(path.relative_to(a.output)):digest(path) for path in a.output.rglob('*') if path.is_file()}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Prepared: {a.output}')


if __name__=='__main__':main()
