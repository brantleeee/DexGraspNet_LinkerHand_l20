"""Small CPU algorithm counterexamples and actual generation geometry audit."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'grasp_generation'))
from admit_l20_middle_contact import load_hand
from robust_surface_query import nearest_surface


def balance_counterexample():
    # Four slots, but only two opposing physical point locations and normal rays.
    p=torch.tensor([[[-.01,0,0],[-.01,0,0],[.01,0,0],[.01,0,0]]])
    n=torch.tensor([[[-1.,0,0],[-1.,0,0],[1.,0,0],[1.,0,0]]])
    transform=p.new_tensor([[0,0,0,0,0,-1,0,1,0],[0,0,1,0,0,0,-1,0,0],[0,-1,0,1,0,0,0,0,0]])
    g=torch.cat([torch.eye(3).expand(1,4,3,3).reshape(1,12,3),(p@transform).reshape(1,12,3)],dim=2)
    fc=torch.norm(n.reshape(1,1,12)@g,dim=(1,2)).square()
    wrench=torch.cat([n[0],torch.cross(p[0],n[0],dim=-1)],dim=-1)
    return dict(E_fc=float(fc),unique_positions=2,selected_slots=4,normal_ray_wrench_rank=int(torch.linalg.matrix_rank(wrench)),
        torque_rays=wrench[:,3:].tolist(),interpretation='Exact E_fc algebra gives zero despite rank-1 normal-ray wrenches; not a friction-cone force-closure or 600g load test.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new directory')
    a.output.mkdir(parents=True);hand,urdf=load_hand();rng=np.random.default_rng(1701);geometry=[]
    for link in ['middle_proximal','middle_middle','middle_distal','base_link']:
        m=hand.mesh[link];mesh=trimesh.Trimesh(m['vertices'].numpy(),m['faces'].numpy(),process=True);hull=mesh.convex_hull
        record=dict(link=link,raw_faces=len(mesh.faces),generation_collision_faces=len(m['face_verts']),convex_hull_faces=len(hull.faces),raw_closed=bool(mesh.is_volume),
            raw_volume_m3=float(mesh.volume),convex_hull_volume_m3=float(hull.volume))
        if mesh.is_volume:
            center=mesh.bounds.mean(0);raw_mm=trimesh.Trimesh((mesh.vertices-center)*1000,mesh.faces,process=False);hull_mm=trimesh.Trimesh((hull.vertices-center)*1000,hull.faces,process=False)
            probes=rng.uniform(mesh.bounds[0],mesh.bounds[1],size=(4096,3));inside_hull=hull_mm.contains((probes-center)*1000);inside_raw=raw_mm.contains((probes-center)*1000)
            only=inside_hull&~inside_raw;record.update(probes=4096,hull_only_points=int(only.sum()))
            if only.any():
                samples=probes[only][:64];q,gap,_=nearest_surface(mesh,samples);j=int(gap.argmax())
                record.update(example_point_m=samples[j].tolist(),example_distance_outside_raw_mm=float(gap[j]*1000))
        geometry.append(record);print(link,record.get('hull_only_points','open source'),flush=True)
    examples=balance_counterexample();result=dict(balance_counterexample=examples,geometry=geometry,
        scope='Static code semantics and deterministic CPU probes, no new generation/training or dynamics',
        code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'grasp_generation/utils/hand_model.py',ROOT/'grasp_generation/utils/energy.py',ROOT/'grasp_generation/utils/optimizer.py',ROOT/'thirdparty/TorchSDF/torchsdf/sdf.py']})
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('Complete:',a.output)


if __name__=='__main__':main()
