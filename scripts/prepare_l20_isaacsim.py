"""Prepare portable full-resolution L20 meshes and saved grasps for Isaac Sim.

No convex hull or decomposition operation is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from l20_object_catalog import OBJECTS

ROOT = Path(__file__).resolve().parents[1]


def xyz(node, key, default='0 0 0'):
    return [float(v) for v in (node.get(key, default) if node is not None else default).split()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', choices=OBJECTS, default='jenga')
    p.add_argument('--result-file', type=Path)
    p.add_argument('--object-mesh', type=Path, help='Original mesh, in the same normalized coordinates as grasp generation')
    p.add_argument('--indices', type=int, nargs='+', help='Indices in the saved NPY, not the scale subgroup')
    p.add_argument('--scale', type=float, default=.06)
    p.add_argument('--density', type=float, default=500.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output must be a new directory')
    spec=OBJECTS[a.object];code=spec['code']
    if a.result_file is None and spec['default_result'] is None:
        p.error(f'{a.object} requires --result-file from freshly generated L20 grasps; bundled legacy poses do not satisfy the current URDF')
    source = (a.result_file or ROOT/spec['default_result']).resolve()
    raw = np.load(source, allow_pickle=True)
    indices = a.indices if a.indices is not None else [i for i,g in enumerate(raw) if abs(float(g['scale'])-a.scale)<1e-6]
    if not indices or any(i<0 or i>=len(raw) for i in indices):
        p.error('No candidates or invalid indices')
    if any(abs(float(raw[i]['scale'])-a.scale)>1e-6 for i in indices):
        p.error('Candidate scale differs from --scale; wrist position cannot be reused across scales')
    urdf = ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'
    robot = ET.parse(urdf).getroot()
    # Share the existing URDF feasibility projection without importing MuJoCo code.
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from validate_grasp_dynamics import project_pose
    candidates=[]
    for i in indices:
        if not np.isfinite(list(raw[i]['qpos'].values())).all():p.error(f'Nonfinite pose at candidate {i}')
        q,feasibility=project_pose(robot,raw[i]['qpos'])
        if spec['default_result'] is None and not feasibility['source_pose_feasible']:
            p.error(f'{a.object} candidate {i} violates the current URDF; regenerate L20 grasps instead of projecting legacy poses')
        candidates.append(dict(index=i,scale=float(raw[i]['scale']),qpos={k:float(v) for k,v in raw[i]['qpos'].items()},
                               targets=q,feasibility=feasibility))
    a.output.mkdir(parents=True)
    arrays, links, provenance = {}, [], []

    def mesh_data(path, key, scale=1.):
        mesh = trimesh.load(path, force='mesh', process=True)
        mesh.vertices *= scale
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(f'Nonfinite mesh: {path}')
        arrays[key+'_vertices'] = np.asarray(mesh.vertices, dtype=np.float32)
        arrays[key+'_faces'] = np.asarray(mesh.faces, dtype=np.int32)
        provenance.append(dict(key=key, path=str(path), sha256=digest(path),
                               vertices=len(mesh.vertices), faces=len(mesh.faces),
                               watertight=bool(mesh.is_watertight), applied_scale=scale))
        return mesh

    for n, link in enumerate(robot.findall('link')):
        inert = link.find('inertial')
        inertia = inert.find('inertia')
        colliders = []
        for k, c in enumerate(link.findall('collision')):
            m = c.find('geometry/mesh')
            key = f'hand_{n}_{k}'
            mesh_data((urdf.parent/m.get('filename')).resolve(), key)
            colliders.append(dict(key=key, xyz=xyz(c.find('origin'),'xyz'), rpy=xyz(c.find('origin'),'rpy')))
        links.append(dict(name=link.get('name'), mass=float(inert.find('mass').get('value')),
                          com=xyz(inert.find('origin'),'xyz'), inertia_rpy=xyz(inert.find('origin'),'rpy'),
                          inertia={k:float(v) for k,v in inertia.attrib.items()}, colliders=colliders))
    joints=[]
    for j in robot.findall('joint'):
        lim=j.find('limit'); mimic=j.find('mimic')
        joints.append(dict(name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'),
                           child=j.find('child').get('link'), xyz=xyz(j.find('origin'),'xyz'),
                           rpy=xyz(j.find('origin'),'rpy'), axis=xyz(j.find('axis'),'xyz','1 0 0'),
                           limit={k:float(v) for k,v in lim.attrib.items()} if lim is not None else {},
                           mimic=dict(mimic.attrib) if mimic is not None else None))
    objpath=(a.object_mesh or ROOT/'data/meshdata'/code/'coacd/decomposed.obj').resolve()
    obj=mesh_data(objpath,'object',a.scale)
    massprops=obj.mass_properties
    if massprops['volume']<=0:
        raise ValueError('Mesh has nonpositive volume; provide an oriented closed mesh')
    manifest=dict(object=a.object, object_code=code, scale=a.scale, source=str(source), source_sha256=digest(source),
                  urdf=str(urdf), urdf_sha256=digest(urdf), links=links,joints=joints,candidates=candidates,
                  object_mesh_kind='user_supplied' if a.object_mesh else spec.get('mesh_kind','recovered_coacd_mesh_not_original'),
                  object_mass_kg=float(massprops['volume']*a.density),
                  object_com=massprops['center_mass'].tolist(),
                  object_inertia=(massprops['inertia']*a.density).tolist(),
                  object_extents_m=obj.extents.tolist(), meshes=provenance,
                  geometry_note='Full hand STL. Object SDF preserves supplied geometry; old CoACD detail cannot be recovered. No hull/decomposition performed.')
    np.savez_compressed(a.output/'meshes.npz',**arrays)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Prepared {len(candidates)} candidates, {len(links)} links; mass={manifest["object_mass_kg"]:.6f} kg: {a.output}')


if __name__=='__main__':
    main()
