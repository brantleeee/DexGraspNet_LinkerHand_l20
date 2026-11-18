"""L20 per-link contact XYZ: CPU Canvas2D viewer + static SVG, no WebGL."""
import argparse,csv,hashlib,html,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from admit_l20_middle_contact import load_hand
from robust_surface_query import nearest_surface
from contact_frame_display import contact_basis,override_basis
import trimesh
FIELDS=['link','global_id','runtime_id','local_id','x_m','y_m','z_m','x_mm','y_mm','z_mm']


def records_for(link,points,offset):
    return [dict(link=link,global_id=offset+i,runtime_id=offset+i,local_id=i,**{a+'_m':float(v) for a,v in zip('xyz',p)},**{a+'_mm':float(v)*1000 for a,v in zip('xyz',p)}) for i,p in enumerate(points)]


def csv_file(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)


def point_svg(rows,axes,frames=None):
    # Static fallback intentionally plots the point set and link origin, not a mesh proxy.
    points=np.array([[r[a+'_mm'] for a in axes] for r in rows]+[[0.,0.]])
    lo=points.min(0);hi=points.max(0);mid=(lo+hi)/2;span=max(float((hi-lo).max()),1.)
    def xy(p):return 280+(p[0]-mid[0])*400/span,250-(p[1]-mid[1])*400/span
    x,y=xy([0,0]);items=[f'<circle cx="{x}" cy="{y}" r="4" fill="black"/><text x="{x+5}" y="{y-5}">O</text>']
    for r,p in zip(rows,points[:-1]):
        x,y=xy(p);label=f'P{r["global_id"]} ({r["x_mm"]:.3f},{r["y_mm"]:.3f},{r["z_mm"]:.3f})mm'
        items.append(f'<circle cx="{x}" cy="{y}" r="4" fill="red"/><text x="{x+5}" y="{y-5}" font-size="11">{label}</text>')
        if frames:
            frame=frames[r['global_id']]
            for key,color in [('Xc','red'),('Yc','green'),('Zc','blue')]:
                end=np.array([r[c+'_mm'] for c in 'xyz'])+2*np.asarray(frame[key]);px,py=xy([end['xyz'.index(c)] for c in axes])
                items.append(f'<line x1="{x}" y1="{y}" x2="{px}" y2="{py}" stroke="{color}" stroke-width="2"/><text x="{px+3}" y="{py}" font-size="9" fill="{color}">P{r["global_id"]} {key}</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 500"><rect width="560" height="500" fill="#f6f8fb"/><text x="15" y="22">'+''.join(axes).upper()+' 点集投影，单位mm；不含网格</text>'+''.join(items)+'</svg>'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--contact-library',type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    hand,urdf=load_hand();a.output.mkdir(parents=True)
    js=ROOT/'scripts/l20_canvas_coordinates.js';(a.output/'viewer.js').write_text(js.read_text())
    override_path=ROOT/'grasp_generation/configs/l20_middle_user_contact_normals_20261116.json'
    override=json.loads(override_path.read_text())
    for path,h in override['asset_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise ValueError('Normal override asset mismatch')
    library=None;library_meta=None
    if a.contact_library:
        library_meta=json.loads((a.contact_library/'manifest.json').read_text())
        for path,h in library_meta['asset_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise ValueError('Contact library source changed')
        for path,h in library_meta['generated_sha256'].items():
            if hashlib.sha256((a.contact_library/path).read_bytes()).hexdigest()!=h:raise ValueError('Contact library changed')
        library={f['global_id']:f for f in json.loads((a.contact_library/'contact_frames.json').read_text())}
    all_rows=[];summary=[];offset=0;all_frames=[]
    for link,m in hand.mesh.items():
        points=m['contact_candidates'].numpy();rows=records_for(link,points,offset);offset+=len(rows)
        if library is not None:
            rows=[r for r in rows if r['global_id'] in library]
            for r in rows:
                entry=library[r['global_id']];r.update(runtime_id=entry['runtime_id'],local_id=entry['local_id'])
                for axis,value in zip('xyz',entry['origin_P_m']):r[axis+'_m']=float(value);r[axis+'_mm']=float(value)*1000
            points=np.asarray([[r[c+'_m'] for c in 'xyz'] for r in rows]).reshape(-1,3)
        all_rows+=rows
        vertices=m['vertices'].numpy();faces=m['faces'].numpy()
        frames=[]
        if len(points):
            mesh=trimesh.Trimesh(vertices,faces,process=False);q,dist,faceids=nearest_surface(mesh,points)
            for r,fi,cp,gap in zip(rows,faceids,q,dist):
                tri=mesh.triangles[fi];raw=mesh.face_normals[fi];n,t1,t2=contact_basis(raw,tri[1]-tri[0]);source='mesh_facet_unreviewed'
                entry=override['points'].get(str(r['global_id'])) if link==override['coordinate_frame'] else None
                if entry:n,t1,t2=override_basis(entry);source='user_confirmed_contact_direction'
                frames.append(dict(global_id=r['global_id'],parent_link=link,origin_P_m=[r[c+'_m'] for c in 'xyz'],surface_projection_Q_m=cp.tolist(),surface_offset_mm=float(gap*1000),Xc=t1.tolist(),Yc=t2.tolist(),Zc=n.tolist(),geometric_N=raw.tolist(),normal_source=source,convention='Xc=T1, Yc=T2, Zc=N; Xc cross Yc=Zc; origin at original candidate P'))
        if library is not None:frames=[library[r['global_id']] for r in rows]
        all_frames+=frames;by_id={f['global_id']:f for f in frames}
        scene=dict(robot='LinkerHand L20',link=link,coordinate_frame=link,vertices_mm=(vertices*1000).astype(float).tolist(),faces=faces.tolist(),points=[dict(global_id=r['global_id'],local_id=r['local_id'],xyz_mm=[r[c+'_mm'] for c in 'xyz'],frame=dict(axes=[by_id[r['global_id']][k] for k in ['Xc','Yc','Zc']])) for r in rows])
        for point in scene['points']:
            f=by_id[point['global_id']];quality=f.get('patch_quality',{})
            if 'support_triangle_m' in quality:
                point['support_triangle_mm']=(1000*np.asarray(quality['support_triangle_m'])).tolist()
                point['anchor_mm']=(1000*np.asarray(f['surface_projection_Q_m'])).tolist()
                point['surface_warning']=f['review_status']
        # Mesh stays original; candidate positions follow an explicit library.
        (a.output/f'{link}.json').write_text(json.dumps(dict(coordinate_frame=link,points=rows),indent=2));csv_file(a.output/f'{link}.csv',rows)
        (a.output/f'{link}_frames.json').write_text(json.dumps(frames,indent=2))
        for axes in ['xy','xz','yz']:(a.output/f'{link}_{axes}.svg').write_text(point_svg(rows,axes,by_id))
        options='<option value="all">全部点</option>'+''.join(f'<option value="{r["global_id"]}">P{r["global_id"]} / local {r["local_id"]}</option>' for r in rows)
        table='<table><tr>'+''.join('<th>'+x+'</th>' for x in FIELDS)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in FIELDS)+'</tr>' for r in rows)+'</table>'
        caption=f'L20 · {link}局部坐标；每个接触点P均为自身坐标系原点，默认Xc=T1红、Yc=T2绿、Zc=N蓝，右手系；可切换成与link XYZ平行的点坐标系。微面法向尚未核查，中指4个方向采用用户覆盖。原网格/点未改，不作深度遮挡。'
        if library is not None:
            caption='新版接触库：P编号保留旧稳定点号，runtime_id才是新连续索引；蓝Zc是新版法向。自动方向待核查，人工方向单独标记。'+caption.replace('原网格/点未改','原网格未改，点坐标以新版库为准')
            if any('support_triangle_m' in f.get('patch_quality',{}) for f in frames):
                caption+=' 选择单点时橙色显示记录的支撑/参考三角面，紫线连接P与表面Q。人工改向后橙面可能仅为历史参考，不能据此宣称新蓝N垂直橙面或已经外向认证；请看方向来源与质量证据。棱边面选择仍需核查。'
            table+='<h3>新版坐标系质量</h3><table><tr><th>稳定点号</th><th>方向来源</th><th>点位移动mm</th><th>质量证据</th><th>待核查</th></tr>'+''.join(f'<tr><td>P{f["global_id"]}</td><td>{f["normal_source"]}</td><td>{f.get("position_change_mm",0):.6f}</td><td>{html.escape(str(f["patch_quality"]))}</td><td>{f["needs_review"]}</td></tr>' for f in frames)+'</table>'
        if not rows:caption+=' 此link没有接触候选点，并非漏画。'
        page='''<!doctype html><meta charset="utf-8"><style>body{font-family:sans-serif;margin:20px}canvas{width:100%;height:650px;touch-action:none;border:1px solid #ccc}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:5px}button{margin:4px}.svg{width:32%;min-width:300px}</style>'''
        page+=f'<a href="index.html">L20全部link目录</a><h1>{link}每个接触点的坐标系与XYZ</h1><p>{caption}</p><p id="state"></p>'
        page+='<button id="reset">重置3D</button>'+''.join(f'<button data-view="{mode}">{mode}</button>' for mode in ['3D','XY','XZ','YZ'])
        page+='<select id="point">'+options+'</select><label><input id="labels" type="checkbox" checked>XYZ标签</label><label><input id="mesh" type="checkbox" checked>显示网格</label><canvas id="view"></canvas>'
        page=page.replace('<canvas id="view">','<label><input id="frames" type="checkbox" checked>每点坐标系</label><select id="frame-mode"><option value="contact">接触基Xc=T1/Yc=T2/Zc=N</option><option value="parallel">每点平行link XYZ</option></select><label>轴长mm <input id="frame-length" type="number" min="0.1" max="20" step="0.5" value="2"></label><canvas id="view">')
        page+='<h2>无JavaScript也可查看的点集投影</h2>'+''.join(f'<a href="{link}_{axes}.svg"><img class="svg" src="{link}_{axes}.svg"></a>' for axes in ['xy','xz','yz'])
        page+=f'<p><a href="{link}.csv">link CSV</a> · <a href="{link}.json">link JSON</a></p>'+table
        page+='<script id="scene-data" type="application/json">'+json.dumps(scene,separators=(',',':')).replace('<','\\u003c')+'</script><script src="viewer.js"></script>'
        (a.output/f'{link}.html').write_text(page)
        summary.append(dict(link=link,count=len(rows),global_indices=[r['global_id'] for r in rows],vertices=len(vertices),triangles=len(faces)))
        print(link,len(rows),'points',flush=True)
    csv_file(a.output/'all_contact_coordinates.csv',all_rows);(a.output/'all_contact_coordinates.json').write_text(json.dumps(all_rows,indent=2))
    (a.output/'all_contact_frames.json').write_text(json.dumps(all_frames,indent=2))
    sources=[urdf,urdf.parent/'contact_points.json',js,Path(__file__),override_path,ROOT/'scripts/robust_surface_query.py',ROOT/'scripts/contact_frame_display.py']
    if a.contact_library:sources.append(a.contact_library/'manifest.json')
    (a.output/'manifest.json').write_text(json.dumps(dict(robot='LinkerHand L20',renderer='Canvas2D software orthographic + static SVG',webgl_required=False,links=summary,total_points=len(all_rows),contact_library=str(a.contact_library) if a.contact_library else None,removed_stable_ids=library_meta['removed_stable_ids'] if library_meta else [],source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}),indent=2))
    links=(f'<p>新版库：已删除稳定点号{library_meta["removed_stable_ids"]}。P号保持旧编号；runtime_id为新连续索引。自动估计仍待核查。</p>' if library is not None else '')+'<table><tr><th>L20 link局部坐标系</th><th>接触点数</th></tr>'+''.join(f'<tr><td><a href="{s["link"]}.html">{s["link"]}</a></td><td>{s["count"]}</td></tr>' for s in summary)+'</table>'
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>L20每个接触点的坐标系（无需WebGL）</h1><p>默认每个P点都有自己的Xc=T1红、Yc=T2绿、Zc=N蓝，原点为P，方向用所属link系表达；可切换成平行link XYZ的点坐标系。Canvas2D软件旋转及XY/XZ/YZ，另附静态SVG。请保留整个目录以加载viewer.js和SVG。</p><p><a href="all_contact_coordinates.csv">全部坐标CSV</a> · <a href="all_contact_frames.json">全部点坐标系JSON</a></p>'+links)
    print('Complete:',a.output.resolve()/'index.html')


if __name__=='__main__':main()
