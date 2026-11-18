"""CPU original-triangle intersection audit, separate from keypoint repulsion.
Tests sampled static angles; does not certify continuous collision-free motion.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'grasp_generation'))


def triangle_overlap(a,b,tol=1e-9):
    """Vectorized triangle SAT including coplanar separating axes; touch included."""
    ea=np.roll(a,-1,axis=1)-a; eb=np.roll(b,-1,axis=1)-b
    na=np.cross(ea[:,0],ea[:,1]); nb=np.cross(eb[:,0],eb[:,1])
    axes=np.concatenate([na[:,None],nb[:,None],
        np.cross(ea[:,:,None],eb[:,None,:]).reshape(-1,9,3),
        np.cross(na[:,None],ea),np.cross(nb[:,None],eb)],axis=1)
    norm=np.linalg.norm(axes,axis=2); valid=norm>1e-14
    axes=axes/np.maximum(norm[:,:,None],1e-30)
    pa=np.einsum('nvc,nac->nav',a,axes);pb=np.einsum('nvc,nac->nav',b,axes)
    separated=((pa.max(2)<pb.min(2)-tol)|(pb.max(2)<pa.min(2)-tol))&valid
    return ~separated.any(1)


def intersect_pairs(a,b):
    from rtree import index
    tree=index.Index(properties=index.Property(dimension=3))
    for i,t in enumerate(b):tree.insert(i,tuple(np.r_[t.min(0),t.max(0)]))
    hits=[]; tested=0
    for i,t in enumerate(a):
        ids=list(tree.intersection(tuple(np.r_[t.min(0)-1e-9,t.max(0)+1e-9])))
        if not ids:continue
        tested+=len(ids)
        for start in range(0,len(ids),2048):
            part=ids[start:start+2048]
            yes=triangle_overlap(np.broadcast_to(t,(len(part),3,3)),b[part])
            hits.extend((i,part[j]) for j in np.flatnonzero(yes))
    return hits,tested


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-file',type=Path,default=ROOT/'data/experiments/l20_clean_drill_20261115/results/scales/scale_1/power-drill-clean-l20.npy')
    p.add_argument('--candidates',nargs='+',type=int,default=[22,18])
    p.add_argument('--angles',nargs='+',type=float,default=[0,.5,.6,.7,.9,1.,1.1])
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    if not np.isfinite(args.angles).all() or min(args.angles)<0 or max(args.angles)>1.4:p.error('MCP angles must be within [0,1.4]')
    from diagnose_l20_middle_objective import pose_from_record
    from utils.hand_model import HandModel
    import torch,hashlib
    urdf=ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'
    hand=HandModel(model_path=str(urdf),mesh_path=str(urdf.parent/'meshes'),contact_points_path=str(urdf.parent/'contact_points.json'),penetration_points_path=str(urdf.parent/'penetration_points.json'),device='cpu')
    raw=np.load(args.result_file,allow_pickle=True)
    if any(i<0 or i>=len(raw) for i in args.candidates):p.error('Invalid source index')
    args.output.mkdir(parents=True)
    config=dict(source=str(args.result_file.resolve()),source_sha256=hashlib.sha256(args.result_file.read_bytes()).hexdigest(),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),angles=args.angles,candidates=args.candidates,tolerance_m=1e-9,algorithm='AABB broad phase + triangle SAT, coplanar axes, touch included',scope='Original visual triangles, no convex hull; static sampled poses only')
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    rows=[]; pages=[]
    targets=['base_link']+[f+'_'+l for f in ['index','ring'] for l in ['metacarpals','proximal','middle','distal']]
    for candidate in args.candidates:
        pose,_=pose_from_record(raw[candidate],hand,'cpu')
        for angle in sorted(set([0.]+args.angles)):
            moved=pose.clone();moved[0,9+hand.joints_names.index('middle_mcp_pitch')]=angle
            hand.set_parameters(hand.joint_constraints.project(moved))
            meshes={}
            for name,m in hand.mesh.items():
                frame=hand.current_status[name].get_matrix()[0].detach().numpy()
                # Local hand frame is sufficient: rigid wrist transform preserves intersection.
                v=m['vertices'].numpy()@frame[:3,:3].T+frame[:3,3]
                meshes[name]=trimesh.Trimesh(v,m['faces'].numpy(),process=False)
            flagged=set(); local=[]
            for middle in ['middle_proximal','middle_middle','middle_distal']:
                for target in targets:
                    a,b=meshes[middle],meshes[target]
                    disjoint=bool(np.any(a.bounds[1]<b.bounds[0]-1e-9) or np.any(b.bounds[1]<a.bounds[0]-1e-9))
                    hits,tested=([],0) if disjoint else intersect_pairs(a.triangles,b.triangles)
                    row=dict(candidate=candidate,mcp_rad=angle,middle_link=middle,target_link=target,
                             surface_intersection_or_touch=bool(hits),triangle_pair_count=len(hits),broad_phase_pairs=tested)
                    # With no surface intersection, test all source vertices for containment
                    # only where the target is a valid closed positive-volume surface.
                    containment=False;eligible=0
                    if not hits and not disjoint:
                        for source,dest in [(a,b),(b,a)]:
                            cleaned=trimesh.Trimesh(dest.vertices,dest.faces,process=True)
                            if cleaned.is_watertight and cleaned.is_winding_consistent and cleaned.volume>0:
                                eligible+=1
                                # Without surface crossings, one vertex per connected
                                # component determines containment; avoid huge ray batches.
                                if 'component_representatives' not in source.metadata:
                                    from scipy.sparse import coo_matrix
                                    from scipy.sparse.csgraph import connected_components
                                    edges=source.edges_unique
                                    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(source.vertices),len(source.vertices)))
                                    _,labels=connected_components(graph,directed=False)
                                    source.metadata['component_representatives']=source.vertices[np.unique(labels,return_index=True)[1]]
                                containment |= bool(cleaned.contains(source.metadata['component_representatives']).any())
                    row.update(disjoint_bounds=disjoint,containment_detected=containment,containment_directions_checked=eligible)
                    rows.append(row);local.append(row)
                    if hits or containment:flagged.update([middle,target])
            import plotly.graph_objects as go
            fig=go.Figure()
            for name in ['base_link']+targets[1:]+['middle_proximal','middle_middle','middle_distal']:
                m=meshes[name];v=m.vertices;f=m.faces
                fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],name=name,showlegend=True,opacity=.6,color='red' if name in flagged else 'royalblue'))
            label=f'candidate_{candidate:03d}_mcp_{angle:.3f}.html'
            fig.update_layout(scene_aspectmode='data',title=f'{candidate}: MCP={angle:.3f} rad; red=intersection/touch or containment; original meshes, no dynamics')
            fig.write_html(args.output/label,include_plotlyjs=True)
            pages.append((candidate,angle,label,[r for r in local if r['surface_intersection_or_touch'] or r['containment_detected']]))
            print(f'{candidate} MCP={angle:.3f}: {len(pages[-1][3])} flagged link pairs',flush=True)
    (args.output/'results.json').write_text(json.dumps(rows,indent=2))
    with (args.output/'pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 中指原网格自碰撞检查','', '红色表示三角面相交/接触或封闭目标包含。不是穿透深度或动力学失败标签；0角已有的结构交叠须单独判断。开放目标的未检出不能认证无实体碰撞；离散角度也不认证连续路径。','', '|候选|MCP rad|检出连杆对|','|---|---:|---|']
    for candidate,angle,label,hits in pages:
        desc='; '.join(r['middle_link']+' / '+r['target_link'] for r in hits) or '未检出（见包含检查范围）'
        lines.append(f'|{candidate}|{angle:.3f}|{desc}|')
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import html
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><pre>'+html.escape('\n'.join(lines))+'</pre>'+''.join(f'<p><a href="{label}">{c} MCP={a:.3f}</a></p>' for c,a,label,_ in pages))
    print('Complete:',args.output.resolve())


if __name__=='__main__':main()
