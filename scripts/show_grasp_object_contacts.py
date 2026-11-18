"""Add original-mesh proximity markers to existing Canvas grasp pages, with backup."""
import argparse,csv,hashlib,html,json,re,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'grasp_generation'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--threshold-mm',type=float,default=1.);a=p.parse_args()
    if not np.isfinite(a.threshold_mm) or a.threshold_mm<=0:p.error('Invalid threshold')
    run=a.run.resolve();view=run/'results/scales/scale_1/top_16_canvas';backup=view.with_name(view.name+'_before_contacts')
    if backup.exists():p.error('Backup exists; refusing repeat in-place modification')
    ranking=json.loads((view/'ranking.json').read_text());source=Path(ranking['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=ranking['source_sha256']:p.error('Result changed')
    import torch,trimesh,transforms3d
    from utils.hand_model import HandModel
    from utils.contact_library import attach_library,world_contact_normals
    from contact_frame_display import contact_basis
    from robust_surface_query import nearest_surface
    library=run/'contact_library';frames=json.loads((library/'contact_frames.json').read_text())
    hand=HandModel(model_path=str(ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'),mesh_path=str(ROOT/'grasp_generation/mjcf/meshes'),contact_points_path=str(library/'contact_points.json'),penetration_points_path=str(ROOT/'grasp_generation/mjcf/penetration_points.json'),n_surface_points=0,device='cpu')
    attach_library(hand,library);data=np.load(source,allow_pickle=True)
    obj=trimesh.load(ROOT/'data/meshdata/power-drill-clean-l20/coacd/decomposed.obj',force='mesh',process=False)
    prepared=[]
    for row in ranking['rows']:
        g=data[row['candidate_index']];q=g['qpos'];r=transforms3d.euler.euler2mat(*[q['WRJR'+c] for c in 'xyz'])
        pose=[q['WRJT'+c] for c in 'xyz']+r[:,:2].T.ravel().tolist()+[q[n] for n in hand.joints_names]
        with torch.no_grad():
            hand.set_parameters(torch.tensor([pose],dtype=torch.float),torch.arange(len(frames))[None]);points=hand.contact_points[0].numpy();normals=world_contact_normals(hand)[0].numpy()
        mesh=obj.copy();mesh.vertices*=float(g['scale']);anchors,gaps,faceids=nearest_surface(mesh,points)
        selected=set(g['contact_point_indices']);records=[];markers=[]
        for i,(f,point,anchor,gap) in enumerate(zip(frames,points,anchors,gaps)):
            near=bool(gap*1000<=a.threshold_mm);chosen=i in selected;n,t1,t2=contact_basis(normals[i],[0,0,1])
            records.append(dict(stable_id=f['stable_id'],runtime_id=i,link=f['parent_link'],selected=chosen,near_object=near,gap_mm=float(gap*1000),hand_world_m=point.tolist(),object_world_m=anchor.tolist(),object_face_id=int(faceids[i])))
            markers.append(dict(global_id=f['stable_id'],xyz_mm=(point*1000).tolist(),object_anchor_mm=(anchor*1000).tolist(),gap_mm=float(gap*1000),near_object=near,selected=chosen,marker_color='#e9a400' if chosen else '#00a9bc',frame=dict(axes=[t1.tolist(),t2.tolist(),n.tolist()])))
        path=view/row['html'];page=path.read_text();match=re.search(r'(<script id="scene-data" type="application/json">)(.*?)(</script>)',page)
        scene=json.loads(match.group(2));scene['points']=markers
        page=page[:match.start(2)]+json.dumps(scene,separators=(',',':'))+page[match.end(2):]
        page=re.sub(r'<select id="point">.*?</select>','<select id="point"><option value="all">接近点与选中点</option>'+''.join(f'<option value="{f["stable_id"]}">P{f["stable_id"]} {f["parent_link"]}</option>' for f in frames)+'</select>',page,count=1)
        page=page.replace('id="frames" type="checkbox" checked','id="frames" type="checkbox"').replace('id="labels" type="checkbox" checked','id="labels" type="checkbox"')
        legend=f'<h2>手—物位置核查</h2><p>青色：距原始物体表面≤{a.threshold_mm:g}mm的手部候选点；黄色：优化器选中点（可能悬空）。绿圈：物体最近表面Q，紫线：P–Q。全部以生成世界系表达。叠加标记不按深度遮挡，便于查看。</p><p>这是静态无符号距离筛查，开放物体网格不能据此判断内外；不是仿真器物理接触或接触力记录。403点中接近{sum(x["near_object"] for x in records)}个，选中唯一点{len(selected)}个。</p><label><input id="near-only" type="checkbox" checked>只显示接近点及选中点</label><label><input id="object-anchors" type="checkbox" checked>显示物体投影位置</label>'
        page=page.replace('<p id="state"></p>',legend+'<p id="state"></p>')
        table='<h2>接近点及选中点坐标（世界系，mm）</h2><table border="1"><tr><th>P号</th><th>link</th><th>选中</th><th>距离mm</th><th>手部P XYZ</th><th>物体Q XYZ</th></tr>'
        for x in records:
            if x['near_object'] or x['selected']:
                table+=f'<tr><td>{x["stable_id"]}</td><td>{x["link"]}</td><td>{x["selected"]}</td><td>{x["gap_mm"]:.4f}</td><td>{np.round(np.array(x["hand_world_m"])*1000,3).tolist()}</td><td>{np.round(np.array(x["object_world_m"])*1000,3).tolist()}</td></tr>'
        page+=table+'</table>';prepared.append((path,page,records))
        print('candidate',row['candidate_index'],'near',sum(x['near_object'] for x in records),'selected',len(selected),flush=True)
    shutil.copytree(view,backup)
    for path,page,records in prepared:
        path.write_text(page);path.with_suffix('.contacts.json').write_text(json.dumps(records,indent=2)+'\n')
    shutil.copyfile(ROOT/'scripts/l20_canvas_coordinates.js',view/'viewer.js')
    (view/'contact_overlay.json').write_text(json.dumps(dict(threshold_mm=a.threshold_mm,backup=str(backup),source_sha256=ranking['source_sha256'],method='unsigned nearest point on original object triangles; no dynamics contact claim'),indent=2)+'\n')
    print('Updated:',view/'index.html')


if __name__=='__main__':main()
