"""Export ranked L20 poses using Canvas2D; display clustering never affects optimization."""
import argparse,json,sys,hashlib,html,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'grasp_generation'))
from export_top_grasps import rank_candidates


def display_mesh(vertices,faces,cell=.0006):
    """Deterministic vertex clustering for software rendering only (metres)."""
    v=np.asarray(vertices);f=np.asarray(faces);_,inverse=np.unique(np.floor(v/cell).astype(np.int64),axis=0,return_inverse=True)
    count=np.bincount(inverse);out=np.column_stack([np.bincount(inverse,weights=v[:,i])/count for i in range(3)])
    ids=inverse[f];keep=(ids[:,0]!=ids[:,1])&(ids[:,1]!=ids[:,2])&(ids[:,0]!=ids[:,2]);ids=ids[keep]
    _,unique=np.unique(np.sort(ids,axis=1),axis=0,return_index=True)
    return out,ids[np.sort(unique)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result-file',required=True,type=Path)
    p.add_argument('--object-code',required=True);p.add_argument('--contact-library',required=True,type=Path)
    p.add_argument('--top-k',default=16,type=int);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    import torch,transforms3d,trimesh
    from utils.hand_model import HandModel
    from utils.contact_library import attach_library,world_contact_normals
    from utils.grasp_results import verify_mesh_identity
    from contact_frame_display import contact_basis
    meshpath=ROOT/'data/meshdata'/a.object_code/'coacd/decomposed.obj';verify_mesh_identity(a.result_file,meshpath)
    data=np.load(a.result_file,allow_pickle=True);order=rank_candidates(data,a.top_k)
    hand=HandModel(model_path=str(ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'),mesh_path=str(ROOT/'grasp_generation/mjcf/meshes'),
        contact_points_path=str(a.contact_library/'contact_points.json'),penetration_points_path=str(ROOT/'grasp_generation/mjcf/penetration_points.json'),n_surface_points=0,device='cpu',force_mesh_sdf=True)
    provenance=attach_library(hand,a.contact_library)
    obj=trimesh.load(meshpath,force='mesh',process=False)
    display={link:display_mesh(m['vertices'].numpy(),m['faces'].numpy()) for link,m in hand.mesh.items()}
    object_display=display_mesh(obj.vertices,obj.faces);a.output.mkdir(parents=True)
    shutil.copyfile(ROOT/'scripts/l20_canvas_coordinates.js',a.output/'viewer.js');rows=[]
    for rank,idx in enumerate(order,1):
        g=data[idx]
        if g['contact_library_manifest_sha256']!=provenance['manifest_sha256']:p.error('Pose and contact library identity mismatch')
        q=g['qpos'];r=transforms3d.euler.euler2mat(*[q['WRJR'+c] for c in 'xyz'])
        pose=[q['WRJT'+c] for c in 'xyz']+r[:,:2].T.ravel().tolist()+[q[n] for n in hand.joints_names]
        contacts=torch.tensor([g['contact_point_indices']],dtype=torch.long)
        with torch.no_grad():
            hand.set_parameters(torch.tensor([pose],dtype=torch.float),contacts)
            normals=world_contact_normals(hand)[0].numpy();cp=hand.contact_points[0].numpy()
        verts=[];faces=[];colors=[];offset=0
        for link,(v,f) in display.items():
            t=hand.current_status[link].get_matrix()[0].numpy();v=(v@t[:3,:3].T+t[:3,3])@r.T+np.array(pose[:3])
            verts.extend((v*1000).tolist());faces.extend((f+offset).tolist());colors.extend(['#8dbbd7']*len(f));offset+=len(v)
        v,f=object_display;verts.extend((v*float(g['scale'])*1000).tolist());faces.extend((f+offset).tolist());colors.extend(['#e6a146']*len(f))
        points=[]
        for j,stable in enumerate(g['contact_stable_ids']):
            n,t1,t2=contact_basis(normals[j],[0,0,1]);points.append(dict(global_id=stable,xyz_mm=(cp[j]*1000).tolist(),frame=dict(axes=[t1.tolist(),t2.tolist(),n.tolist()])))
        scene=dict(link='生成世界坐标系',vertices_mm=verts,faces=faces,face_colors=colors,points=points)
        title=f'电钻 L20 | 能量排名{rank} | 原下标{idx} | scale={g["scale"]:g}'
        controls='<button id="reset">重置</button>'+''.join(f'<button data-view="{x}">{x}</button>' for x in ['3D','XY','XZ','YZ'])
        controls+='<select id="point"><option value="all">本姿态全部选中点</option>'+''.join(f'<option value="{x["global_id"]}">P{x["global_id"]}</option>' for x in points)+'</select>'
        controls+=''.join(f'<label><input id="{x}" type="checkbox" checked>{label}</label>' for x,label in [('labels','点标签'),('mesh','模型'),('frames','每点坐标系')])
        controls+='<select id="frame-mode"><option value="contact">接触框架</option><option value="parallel">世界平行轴</option></select><input id="frame-length" type="number" value="6" min="0.1" max="20"><p id="state"></p><canvas id="view"></canvas>'
        page='<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;margin:20px}canvas{width:100%;height:650px;touch-action:none}button,label{margin:5px}</style>'
        page+=f'<a href="index.html">排名目录</a><h1>{html.escape(title)}</h1><p>蓝色手，橙色电钻；红/绿为切向，接触法向蓝Zc。仅静态生成，未做动力学。为Canvas2D交互，显示网格采用0.6mm顶点聚类；优化和碰撞原网格未改变。P为最终选中稳定点号，不是旧数组索引。</p><p>总能量={g["energy"]:.6g}；E_dis={g["E_dis"]:.6g} m；E_fc={g["E_fc"]:.6g}；E_normal={g["E_normal"]:.6g}</p>'+controls
        page+='<script id="scene-data" type="application/json">'+json.dumps(scene,separators=(',',':'))+'</script><script src="viewer.js"></script>'
        name=f'rank_{rank:02d}_candidate_{idx:03d}.html';(a.output/name).write_text(page)
        rows.append(dict(rank=rank,candidate_index=idx,energy=float(g['energy']),E_normal=float(g['E_normal']),stable_contact_ids=g['contact_stable_ids'],html=name))
    (a.output/'ranking.json').write_text(json.dumps(dict(source=str(a.result_file.resolve()),source_sha256=hashlib.sha256(a.result_file.read_bytes()).hexdigest(),contact_provenance=provenance,display_only_cell_m=.0006,rows=rows),indent=2)+'\n')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>L20 v6 电钻静态生成 · 前'+str(a.top_k)+' · 无WebGL</h1><p>按含法向项的总能量排名；不是动力学成功排名。显示聚类不用于优化。</p><ol>'+''.join(f'<li><a href="{x["html"]}">候选{x["candidate_index"]}，能量{x["energy"]:.6g}</a></li>' for x in rows)+'</ol>')
    print(a.output/'index.html',flush=True)


if __name__=='__main__':main()
