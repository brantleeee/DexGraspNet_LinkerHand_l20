"""Audit saved v2 poses and SAME sample cloud against independent closed-mesh rays."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,torch,trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'grasp_generation'))
from utils.hand_model_v2 import HandModel
from torchsdf import compute_sdf
from diagnose_l20_middle_objective import pose_from_record
from robust_surface_query import nearest_surface


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'data/experiments/l20_v2_clean_drill_20261117_174021');p.add_argument('--candidates',nargs='+',type=int,default=[13,5,18]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    r=a.run;source=r/'results/scales/scale_1/power-drill-clean-l20.npy';data=np.load(source,allow_pickle=True);cloud=np.load(r/'object_surface_points.npy');urdf=ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'
    hand=HandModel(model_path=str(urdf),mesh_path=str(urdf.parent/'meshes'),contact_points_path=str(r/'contact_library/contact_points.json'),device='cuda',penetration_geometry='raw-mesh')
    meshes={}
    for link,m in hand.mesh.items():
        mesh=trimesh.Trimesh(m['vertices'].cpu().numpy(),m['faces'].cpu().numpy(),process=True)
        if mesh.is_volume:meshes[link]=mesh
    a.output.mkdir(parents=True);reports=[]
    for index in a.candidates:
        pose,_=pose_from_record(data[index],hand,'cuda');hand.set_parameters(pose)
        points=torch.tensor(cloud[index:index+1]*float(data[index]['scale']),device='cuda')
        with torch.no_grad():signed=hand.cal_distance(points);value=float(signed.clamp_min(0).sum())
        report=dict(candidate=index,saved_E_pen=float(data[index]['E_pen']),recomputed_E_pen=value,closed_link_sign_checks=[])
        report['saved_score_reproduced']=bool(np.isclose(value,report['saved_E_pen'],atol=2e-4,rtol=1e-3))
        print(index,'stored',report['saved_E_pen'],'recomputed',value,flush=True)
        local_hand=(points-hand.global_translation[:,None])@hand.global_rotation
        for link,mesh in meshes.items():
            t=hand.current_status[link].get_matrix();local=((local_hand-t[:,:3,3][:,None])@t[:,:3,:3])[0]
            with torch.no_grad():d,s,_,_=compute_sdf(local,hand.mesh[link]['face_verts']);cuda=(d+1e-8).sqrt()*(-s)
            xyz=local.cpu().numpy();bounds=(xyz>=mesh.bounds[0]).all(-1)&(xyz<=mesh.bounds[1]).all(-1)
            if not bounds.any():continue
            ids=np.flatnonzero(bounds);center=mesh.bounds.mean(0);scaled=trimesh.Trimesh((mesh.vertices-center)*1000,mesh.faces,process=False)
            inside=scaled.contains((xyz[ids]-center)*1000);gpu_inside=cuda[ids].cpu().numpy()>0
            missed=ids[inside&~gpu_inside];false=ids[~inside&gpu_inside]
            row=dict(link=link,queries_in_bounds=len(ids),ray_inside=int(inside.sum()),cuda_inside=int(gpu_inside.sum()),false_negative_indices=missed.tolist(),false_positive_indices=false.tolist())
            if len(missed):
                sample=missed[:64];_,gaps,_=nearest_surface(mesh,xyz[sample]);j=int(gaps.argmax());row.update(missed_example_sample=int(sample[j]),missed_example_raw_surface_distance_mm=float(gaps[j]*1000),cuda_signed_at_example_mm=float(cuda[sample[j]].cpu()*1000))
            report['closed_link_sign_checks'].append(row)
        reports.append(report);(a.output/'results.json').write_text(json.dumps(reports,indent=2)+'\n');print(index,'same Epen',value,'missed rays',sum(len(x['false_negative_indices']) for x in report['closed_link_sign_checks']),flush=True)
    (a.output/'config.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),same_saved_surface_samples=True,note='Closed-mesh ray comparison only; not an independent physical solid certificate for overlapping components/open meshes.'),indent=2)+'\n')


if __name__=='__main__':main()
