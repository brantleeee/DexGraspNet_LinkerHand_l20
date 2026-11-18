"""Official ShadowHand: every link and every local contact XYZ. CPU only."""
import argparse,csv,hashlib,html,json,os,sys
from pathlib import Path
import numpy as np
import trimesh
from contact_frame_display import add_link_axes
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external/DexGraspNet_official/grasp_generation'
FIELDS=['link','global_id','local_id','x_m','y_m','z_m','x_mm','y_mm','z_mm']


def coordinate_record(link,gid,lid,point):
    return dict(link=link,global_id=gid,local_id=lid,**{axis+'_m':float(x) for axis,x in zip('xyz',point)},**{axis+'_mm':float(x)*1000 for axis,x in zip('xyz',point)})


def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--label-unit',choices=['m','mm'],default='mm');a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    sys.path.insert(0,str(SOURCE))
    from utils.hand_model import HandModel
    import utils.hand_model as module
    if Path(module.__file__).resolve()!=SOURCE/'utils/hand_model.py':raise ValueError('Wrong HandModel imported')
    old=Path.cwd()
    try:
        os.chdir(SOURCE)
        hand=HandModel(mjcf_path='mjcf/shadow_hand_wrist_free.xml',mesh_path='mjcf/meshes',contact_points_path='mjcf/contact_points.json',penetration_points_path='mjcf/penetration_points.json',device='cpu')
    finally:os.chdir(old)
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    a.output.mkdir(parents=True);(a.output/'plotly.min.js').write_text(get_plotlyjs())
    links=[]
    def walk(frame):
        if frame.link.name not in links:links.append(frame.link.name)
        for child in frame.children:walk(child)
    walk(hand.chain._root)
    # Preserve the actual official global contact ordering, independent of empty frames.
    points_by_link={};offset=0
    for name,data in hand.mesh.items():
        values=data['contact_candidates'].numpy();points_by_link[name]=(values,offset);offset+=len(values)
    all_rows=[];summary=[]
    for number,link in enumerate(links):
        filename=f'link_{number:03d}_{link.replace("robot0:", "")}.html'
        fig=go.Figure();data=hand.mesh.get(link);geometry='no_visual_geometry_virtual_frame'
        if data is not None:
            if 'geom_param' in data:
                radius=float(data['geom_param'][0]);height=2*float(data['geom_param'][1])
                mesh=trimesh.creation.capsule(radius=radius,height=height)
                mesh.apply_translation(np.array([0,0,height/2])-mesh.bounds.mean(0))
                geometry='analytic_capsule_display_matching_official_cal_distance'
            else:
                mesh=trimesh.Trimesh(data['vertices'].numpy(),data['faces'].numpy(),process=False)
                geometry='official_non_capsule_runtime_mesh'
            v=mesh.vertices;f=mesh.faces
            fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='lightgray',opacity=.4,name=link))
        add_link_axes(fig,link);base=list(range(len(fig.data)));point_traces=[];rows=[]
        points,first=points_by_link.get(link,(np.empty((0,3)),0))
        for i,point in enumerate(points):
            gid=first+i;r=coordinate_record(link,gid,i,point);rows.append(r);all_rows.append(r)
            xyz=point*1000 if a.label_unit=='mm' else point
            decimals=3 if a.label_unit=='mm' else 6
            label=f'P{gid} ({xyz[0]:.{decimals}f}, {xyz[1]:.{decimals}f}, {xyz[2]:.{decimals}f}) {a.label_unit}'
            hover=f'{html.escape(link)}<br>global={gid}, local={i}<br>x={r["x_m"]:.9f} m / {r["x_mm"]:.6f} mm<br>y={r["y_m"]:.9f} m / {r["y_mm"]:.6f} mm<br>z={r["z_m"]:.9f} m / {r["z_mm"]:.6f} mm'
            point_traces.append(len(fig.data))
            fig.add_trace(go.Scatter3d(x=[point[0]],y=[point[1]],z=[point[2]],mode='markers+text',text=[label],textposition='top center',marker=dict(size=5,color='red'),name=f'P{gid} / local{i}',hovertemplate=hover+'<extra></extra>'))
        def visibility(ids):return [i in ids for i in range(len(fig.data))]
        buttons=[dict(label='全部点+XYZ',method='update',args=[{'visible':[True]*len(fig.data)}])]
        buttons += [dict(label=f'仅点{first+i}',method='update',args=[{'visible':visibility(base+[t])}]) for i,t in enumerate(point_traces)]
        views=[dict(label=label,method='relayout',args=[{'scene.camera.eye':eye}]) for label,eye in [('＋X',dict(x=2,y=0,z=.2)),('−X',dict(x=-2,y=0,z=.2)),('＋Y',dict(x=0,y=2,z=.2)),('−Y',dict(x=0,y=-2,z=.2)),('＋Z',dict(x=.1,y=0,z=2)),('−Z',dict(x=.1,y=0,z=-2))]]
        fig.update_layout(height=900,margin=dict(t=140),scene_aspectmode='data',scene=dict(xaxis_title='X_link (m)',yaxis_title='Y_link (m)',zaxis_title='Z_link (m)'),title=link+f'：{len(rows)}个接触点；XYZ均为本link局部坐标',updatemenus=[dict(buttons=buttons,x=0,y=1.15),dict(type='buttons',buttons=views,x=.3,y=1.15)])
        caption='坐标系为标题中的官方PK link（含_child/虚拟框架），未转换到手腕或世界。图上XYZ标签单位='+a.label_unit+'，坐标轴始终为米；悬停和表格同时提供m/mm。胶囊显示放在官方解析SDF位置，未更改点坐标。'
        if not rows:caption+=' 此link没有接触候选点；若没有显示几何，仅绘制计算框架坐标轴，不代表遗漏数据。'
        table='<table border="1"><tr>'+''.join('<th>'+x+'</th>' for x in FIELDS)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in FIELDS)+'</tr>' for r in rows)+'</table>'
        stem=filename[:-5];write_csv(a.output/(stem+'.csv'),rows)
        (a.output/(stem+'.json')).write_text(json.dumps(dict(coordinate_frame=link,units='metres in *_m; millimetres in *_mm',points=rows),indent=2))
        (a.output/filename).write_text('<!doctype html><meta charset="utf-8"><a href="index.html">所有link目录</a><p>'+caption+'</p>'+fig.to_html(full_html=False,include_plotlyjs='plotly.min.js')+f'<p><a href="{stem}.csv">CSV</a> · <a href="{stem}.json">JSON</a></p>'+table)
        summary.append(dict(link=link,page=filename,count=len(rows),geometry=geometry,virtual_without_geometry=data is None))
    if len(all_rows)!=hand.n_contact_candidates:raise ValueError('Contact count mismatch')
    write_csv(a.output/'all_contact_coordinates.csv',all_rows)
    (a.output/'all_contact_coordinates.json').write_text(json.dumps(all_rows,indent=2))
    sources=[SOURCE/'mjcf/shadow_hand_wrist_free.xml',SOURCE/'mjcf/contact_points.json',SOURCE/'utils/hand_model.py']
    (a.output/'manifest.json').write_text(json.dumps(dict(links=summary,total_points=len(all_rows),label_unit=a.label_unit,scope='Official link-local point inspection, no simulation or normal corrections',source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}),indent=2))
    def table_for(items):return '<table border="1"><tr><th>link局部系</th><th>点数</th><th>几何</th></tr>'+''.join(f'<tr><td><a href="{r["page"]}">{html.escape(r["link"])}</a></td><td>{r["count"]}</td><td>{r["geometry"]}</td></tr>' for r in items)+'</table>'
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>官方ShadowHand：各link接触点XYZ</h1><p>局部坐标，不是世界坐标。所有页面及plotly.min.js请保存在同一目录。</p><p><a href="all_contact_coordinates.csv">全部点CSV</a> · <a href="all_contact_coordinates.json">全部点JSON</a></p><h2>有接触点的link</h2>'+table_for([r for r in summary if r['count']])+'<h2>无接触点/计算框架</h2>'+table_for([r for r in summary if not r['count']]))
    print(f'Complete: {a.output.resolve()}/index.html; {len(summary)} link frames, {len(all_rows)} points')


if __name__=='__main__':main()
