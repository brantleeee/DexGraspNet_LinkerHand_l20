"""Build an audited closed nonconvex physics proxy for saved clean-surface grasps.

Keeps source grasps immutable; never estimates rigid mass from the open surface.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import trimesh
from prepare_power_drill import close_matching_bore
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_META=Path('/home/lee/robot_learning/pose_estimation/rm_arm_mujoco/assets/objects/power_drill/asset_metadata.json')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def closed_proxy(mesh):
    mesh=mesh.copy();bounds=mesh.bounds.copy()
    mesh.merge_vertices(merge_tex=True,merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces());mesh.update_faces(mesh.unique_faces());mesh.remove_unreferenced_vertices()
    parts=mesh.split(only_watertight=False);solids=[];added=0
    for part in parts:
        if not part.is_watertight:part,n=close_matching_bore(part);added+=n
        trimesh.repair.fix_normals(part,multibody=True)
        if not part.is_volume:raise ValueError('Component is not a closed oriented volume')
        solids.append(part)
    solid=trimesh.boolean.union(solids,engine='manifold',check_volume=True)
    if not solid.is_volume or not np.allclose(solid.bounds,bounds,rtol=0,atol=2e-7):raise ValueError('Proxy invalid or changes object bounds')
    return solid,dict(component_count=len(parts),added_bore_faces=added,watertight=bool(solid.is_watertight),winding_consistent=bool(solid.is_winding_consistent),bounds_max_difference_m=float(abs(solid.bounds-bounds).max()),convex_hull=False,convex_decomposition=False,decimation=False,operations=['weld exact seams','drop degenerate/duplicate faces','bridge matching bore rims preserving bore','manifold solid union'],coordinate_transform='identity; no rotation, translation, normalization or scale')


def prepare(source,out,metadata):
    if out.exists():raise ValueError('Output must be new')
    inputs=source/'inputs';m=json.loads((inputs/'manifest.json').read_text());info=json.loads((source/'mesh_input.json').read_text())
    objpath=source/'source_mesh.obj'
    if sha(objpath)!=info['mesh_sha256'] or info['source_scale']!=1.:raise ValueError('Frozen source changed or not in physical units')
    data=np.load(inputs/'meshes.npz');obj=trimesh.load(objpath,force='mesh',process=False)
    if not np.allclose(obj.vertices,data['object_vertices'],rtol=0,atol=1e-8) or not np.array_equal(obj.faces,data['object_faces']):raise ValueError('Generation surface arrays mismatch')
    solid,audit=closed_proxy(obj)
    meta=json.loads(metadata.read_text());mass=float(meta['mass_kg_assumed']);diag=np.asarray(meta['diaginertia_kg_m2'],float)
    if not np.isfinite(mass) or mass<=0 or diag.shape!=(3,) or not np.isfinite(diag).all() or min(diag)<=0:raise ValueError('Invalid mass/inertia metadata')
    expected=mass/12*np.array([sum(obj.extents[[1,2]]**2),sum(obj.extents[[0,2]]**2),sum(obj.extents[[0,1]]**2)])
    if not np.allclose(expected,diag,rtol=1e-5):raise ValueError('Source bounding-box inertia does not match this object scale')
    out.mkdir(parents=True);target=out/'inputs';target.mkdir()
    collision=out/'collision_closed.obj';solid.export(collision);solid=trimesh.load(collision,force='mesh',process=False)
    shutil.copy2(objpath,out/'generation_surface.obj');shutil.copy2(metadata,out/'physical_metadata.json')
    arrays={n:data[n] for n in data.files};arrays['object_vertices']=np.asarray(solid.vertices,np.float32);arrays['object_faces']=np.asarray(solid.faces,np.int32)
    new=copy.deepcopy(m)
    new.update(object_mesh_kind='closed_nonconvex_proxy_from_frozen_clean_surface',object_mass_kg=mass,object_com=[0.,0.,0.],object_inertia=np.diag(diag).tolist(),mass_estimate_valid=True,mass_is_measured=False,geometry_note='Generated on open clean surface; dynamic collisions use audited bore-repaired nonconvex solid union in identical coordinates. Mass/inertia are source metadata assumptions, not measurements.')
    new['mesh_input']=dict(info,physics_admitted=True,physics_mesh=str(collision),generation_surface_watertight=False,object_watertight=True)
    new['physics_proxy_audit']=dict(audit,source_generation=str(source),source_manifest_sha256=sha(inputs/'manifest.json'),source_arrays_sha256=sha(inputs/'meshes.npz'),generation_surface_sha256=sha(objpath),collision_sha256=sha(collision),physical_metadata_sha256=sha(metadata),mass_kg=mass,inertia_model=meta['inertia_model'],candidates_unchanged=new['candidates']==m['candidates'],hand_model_unchanged=True)
    for mesh in new['meshes']:
        if mesh['key']=='object':mesh.update(path=str(collision),sha256=sha(collision),vertices=len(solid.vertices),faces=len(solid.faces))
    np.savez_compressed(target/'meshes.npz',**arrays);(target/'manifest.json').write_text(json.dumps(new,indent=2))
    (out/'physics_proxy_audit.json').write_text(json.dumps(new['physics_proxy_audit'],indent=2));shutil.copy2(source/'input_audit.json',out/'input_audit.json')
    print(f'Prepared {len(new["candidates"])} unchanged grasps, {len(new["candidates"])*6} directions; repaired {audit["added_bore_faces"]} faces; mass={mass}kg',flush=True)
    subprocess.run([str(ROOT/'.venv_l20/bin/python'),str(ROOT/'scripts/graspqp_drill_pipeline.py'),'render','--output',str(out)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generation',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--physical-metadata',type=Path,default=DEFAULT_META);p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared-run',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.prepared_run:
        if not (out/'physics_proxy_audit.json').exists():p.error('Missing prepared physics proxy')
        m=json.loads((out/'inputs/manifest.json').read_text());audit=json.loads((out/'physics_proxy_audit.json').read_text())
        if sha(out/'collision_closed.obj')!=audit['collision_sha256']:p.error('Prepared physics proxy changed')
        if m['candidates']!=json.loads((a.generation/'inputs/manifest.json').read_text())['candidates']:p.error('Grasp source changed')
    else:prepare(a.generation.resolve(),out,a.physical_metadata.resolve())
    if not a.prepare_only:
        subprocess.run([str(ROOT/'.venv_l20/bin/python'),str(ROOT/'scripts/graspqp_drill_pipeline.py'),'validate','--output',str(out)],cwd=ROOT,check=True)

if __name__=='__main__':main()
