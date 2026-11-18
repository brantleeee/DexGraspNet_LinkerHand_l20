"""Original mesh gap probes for the link pairs driving middle opening; no optimization."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,trimesh
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'grasp_generation'))
import mesh_gap
from diagnose_l20_middle_objective import pose_from_record
raw_triangle_distance=mesh_gap.triangle_distance


def normalized_triangle_distance(a,b):
    # Normalize each pair before old point/segment/SAT operations, which otherwise
    # use absolute tolerances on metre-scale microtriangles.
    origin=a[:,0].copy()
    scale=np.maximum(np.linalg.norm(a-np.roll(a,1,axis=1),axis=2).max(1),np.linalg.norm(b-np.roll(b,1,axis=1),axis=2).max(1))
    if (scale<=1e-15).any():raise ValueError('Degenerate triangle in gap audit')
    return raw_triangle_distance((a-origin[:,None])/scale[:,None,None],(b-origin[:,None])/scale[:,None,None])*scale


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'data/experiments/l20_v6_clean_drill_20261117_135743')
    p.add_argument('--candidates',nargs='+',type=int,default=[21,7]);p.add_argument('--deltas',nargs='+',type=float,default=[0.,.2])
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    from utils.hand_model import HandModel
    import torch
    urdf=ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'
    hand=HandModel(model_path=str(urdf),mesh_path=str(urdf.parent/'meshes'),contact_points_path=str(a.run/'contact_library/contact_points.json'),penetration_points_path=str(urdf.parent/'penetration_points.json'),n_surface_points=0,device='cpu')
    source=a.run/'results/scales/scale_1/power-drill-clean-l20.npy';data=np.load(source,allow_pickle=True)
    mesh_gap.triangle_distance=normalized_triangle_distance
    pairs=[('middle_proximal','index_proximal'),('middle_proximal','ring_proximal'),('middle_distal','ring_middle')]
    a.output.mkdir(parents=True);rows=[]
    for i in a.candidates:
        base,_=pose_from_record(data[i],hand,'cpu')
        for delta in a.deltas:
            pose=base.clone();idx=9+hand.joints_names.index('middle_mcp_pitch');pose[0,idx]+=delta
            hand.set_parameters(hand.joint_constraints.project(pose));meshes={}
            for name in set(sum((list(x) for x in pairs),[])):
                m=hand.mesh[name];t=hand.current_status[name].get_matrix()[0].detach().numpy()
                meshes[name]=trimesh.Trimesh(m['vertices'].numpy()@t[:3,:3].T+t[:3,3],m['faces'].numpy(),process=False)
            for left,right in pairs:
                l,r=meshes[left],meshes[right];gap=mesh_gap.truncated_gap(l,r,.001)
                inside=[mesh_gap.contained(l,r),mesh_gap.contained(r,l)] if gap>1e-9 else [None,None]
                row=dict(candidate=i,delta=delta,mcp_rad=float(pose[0,idx]),left=left,right=right,surface_gap_clipped_mm=gap*1000,
                    clipping_mm=1,containment_checks=inside,detected_surface_touch_or_intersection=bool(gap<=1e-9),
                    detected_containment=any(x is True for x in inside),unknown_containment=any(x is None for x in inside))
                rows.append(row);print(row,flush=True)
                (a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    config=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Original triangle surface gaps clipped at 1mm; isolated static MCP increments; not whole-hand or continuous-path certification',normalized_microtriangles=True)
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    lines=['# 关键点排斥对应连杆的原网格间隙','','1mm表示≥1mm，0表示表面接触/相交；包含未知仍保留，不能据正间隙认证实体完全无碰撞。','', '|候选|MCP增量|连杆对|截断间隙mm|包含未知|','|---|---:|---|---:|---|']
    for r in rows:lines.append(f'|{r["candidate"]}|{r["delta"]}|{r["left"]}/{r["right"]}|{r["surface_gap_clipped_mm"]:.6g}|{r["unknown_containment"]}|')
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Complete:',a.output.resolve())


if __name__=='__main__':main()
