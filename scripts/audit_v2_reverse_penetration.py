"""Independent reverse-direction containment in a documented closed object proxy."""
import argparse,json,sys
from pathlib import Path
import numpy as np,trimesh,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'grasp_generation'))
from utils.hand_model import HandModel
from diagnose_l20_middle_objective import pose_from_record
from validate_clean_graspqp_drill import closed_proxy
from robust_surface_query import nearest_surface


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('New directory required')
    run=ROOT/'data/experiments/l20_v2_clean_drill_20261117_174021';data=np.load(run/'results/scales/scale_1/power-drill-clean-l20.npy',allow_pickle=True)
    surface=trimesh.load(ROOT/'data/meshdata/power-drill-clean-l20/coacd/decomposed.obj',force='mesh',process=False)
    proxy,repair=closed_proxy(surface);center=proxy.bounds.mean(0);mm=trimesh.Trimesh((proxy.vertices-center)*1000,proxy.faces,process=False)
    urdf=ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf';hand=HandModel(model_path=str(urdf),mesh_path=str(urdf.parent/'meshes'),contact_points_path=str(run/'contact_library/contact_points.json'),n_surface_points=0,device='cpu')
    a.output.mkdir(parents=True);proxy.export(a.output/'object_closed_proxy.obj');rows=[]
    for index in [13,5,18]:
        pose,_=pose_from_record(data[index],hand,'cpu');hand.set_parameters(pose);links=[]
        for name,m in hand.mesh.items():
            ids=np.linspace(0,len(m['vertices'])-1,min(512,len(m['vertices']))).astype(int);v=m['vertices'][ids].numpy();t=hand.current_status[name].get_matrix()[0].numpy()
            world=(v@t[:3,:3].T+t[:3,3])@hand.global_rotation[0].numpy().T+hand.global_translation[0].numpy()
            inside=mm.contains((world-center)*1000)
            if inside.any():
                points=world[inside];_,gap,_=nearest_surface(proxy,points);deep=gap>.001
                j=int(gap.argmax());links.append(dict(link=name,tested_vertices=len(ids),inside_vertices=int(inside.sum()),inside_over_1mm=int(deep.sum()),max_inside_mm=float(gap[j]*1000),example_world_m=points[j].tolist()))
        row=dict(candidate=index,saved_one_way_E_pen=float(data[index]['E_pen']),links=links);rows.append(row)
        print(index,'inside vertices',sum(x['inside_vertices'] for x in links),'maxdepth',max(x['max_inside_mm'] for x in links),flush=True)
        (a.output/'results.json').write_text(json.dumps(dict(proxy_repair=repair,candidates=rows,note='Deterministic sampled hand vertices vs same-source repaired closed object; proxy-dependent diagnostic, not dynamics certification.'),indent=2)+'\n')


if __name__=='__main__':main()
