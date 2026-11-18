"""CPU local-frame distal contact inspection; no simulation or pose changes."""
import argparse,hashlib,html,json
from pathlib import Path
import numpy as np
import trimesh
from admit_l20_middle_contact import load_hand
from contact_frame_display import contact_basis,override_basis,add_arrow,add_link_axes
from robust_surface_query import nearest_surface
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normal-overrides',type=Path,default=ROOT/'grasp_generation/configs/l20_middle_user_contact_normals_20261116.json')
    p.add_argument('--vector-mm',type=float,default=2.)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    if not np.isfinite(a.vector_mm) or a.vector_mm<=0:p.error('Invalid vector length')
    overrides=json.loads(a.normal_overrides.read_text())
    for path,h in overrides['asset_hashes'].items():
        if sha(path)!=h:raise ValueError('Override model changed: '+path)
    if sha(overrides['source_frames'])!=overrides['source_frames_sha256']:raise ValueError('Override source frames changed')
    hand,urdf=load_hand();owners=[list(hand.mesh)[i] for i in hand.global_index_to_link_index.tolist()]
    a.output.mkdir(parents=True);summary=[]
    import plotly.graph_objects as go
    for finger in ['thumb','index','middle','ring','pinky']:
        link=finger+'_distal';data=hand.mesh[link];v=data['vertices'].numpy();f=data['faces'].numpy()
        mesh=trimesh.Trimesh(v,f,process=False);ids=[i for i,n in enumerate(owners) if n==link]
        pts=hand.contact_candidates[ids].numpy();closest,distance,faces=nearest_surface(mesh,pts)
        fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='lightgray',opacity=.5,flatshading=True,name=link))
        add_link_axes(fig,link);base=list(range(len(fig.data)));markers=[];normal_traces=[];frames=[];point_groups={}
        for local,(idx,point,cp,gap,fi) in enumerate(zip(ids,pts,closest,distance,faces)):
            tri=mesh.triangles[fi];raw=mesh.face_normals[fi]
            n,t1,t2=contact_basis(raw,tri[1]-tri[0]);source='mesh_facet_unreviewed'
            record=overrides['points'].get(str(idx)) if link==overrides['coordinate_frame'] else None
            if record:
                n,t1,t2=override_basis(record);source='user_confirmed_override'
            group=f'point{idx}';start=len(fig.data);markers.append(start)
            fig.add_trace(go.Scatter3d(x=[point[0]],y=[point[1]],z=[point[2]],text=[str(idx)],mode='markers+text',marker=dict(size=4,color='orange' if record else 'black'),name=f'{idx} (local {local})',legendgroup=group))
            for direction,color,label in [(n,'red','N'),(t1,'green','T1'),(t2,'blue','T2')]:
                before=len(fig.data);add_arrow(fig,cp,direction,a.vector_mm/1000,color,f'{idx} {label}',group)
                if label=='N':normal_traces+=list(range(before,len(fig.data)))
            fig.add_trace(go.Scatter3d(x=[point[0],cp[0]],y=[point[1],cp[1]],z=[point[2],cp[2]],mode='markers+lines',marker=dict(size=2,color='gray'),line=dict(color='gray',dash='dash'),showlegend=False,legendgroup=group,name=f'{idx} P-Q'))
            point_groups[idx]=list(range(start,len(fig.data)))
            bary=trimesh.triangles.points_to_barycentric((tri-cp)[None]*1000,np.zeros((1,3)))[0]
            frames.append(dict(index=idx,local_index=local,coordinate_frame=link,candidate_P_m=point.tolist(),surface_anchor_Q_m=cp.tolist(),distance_to_hand_surface_mm=float(gap*1000),face_id=int(fi),geometric_N=raw.tolist(),N=n.tolist(),T1=t1.tolist(),T2=t2.tolist(),normal_source=source,override_rule=record['rule'] if record else None,projection_on_edge=bool(np.min(np.abs(bary))<1e-5)))
        def visibility(indices):return [i in indices for i in range(len(fig.data))]
        buttons=[dict(label='全部点：仅N',method='update',args=[{'visible':visibility(base+markers+normal_traces)}]),
                 dict(label='全部点：N/T1/T2',method='update',args=[{'visible':[True]*len(fig.data)}]),
                 dict(label='全部点：不显示向量',method='update',args=[{'visible':visibility(base+markers)}])]
        buttons += [dict(label=f'仅点{idx} / local {j}',method='update',args=[{'visible':visibility(base+point_groups[idx])}]) for j,idx in enumerate(ids)]
        views=[dict(label=label,method='relayout',args=[{'scene.camera.eye':eye}]) for label,eye in [('＋X',dict(x=2,y=0,z=.2)),('−X',dict(x=-2,y=0,z=.2)),('＋Y',dict(x=0,y=2,z=.2)),('−Y',dict(x=0,y=-2,z=.2)),('＋Z',dict(x=.1,y=0,z=2)),('−Z',dict(x=.1,y=0,z=-2))]]
        for i,t in enumerate(fig.data):t.visible=i in base+markers+normal_traces
        fig.update_layout(scene_aspectmode='data',scene=dict(xaxis_title='X_link (m)',yaxis_title='Y_link (m)',zaxis_title='Z_link (m)'),title=f'{link}: 原连杆局部坐标；N红 / T1绿 / T2蓝',height=900,legend=dict(groupclick='togglegroup'),updatemenus=[dict(buttons=buttons,x=0,y=1.14),dict(type='buttons',buttons=views,x=.35,y=1.14)])
        caption='全局接触索引用于反馈，local索引仅指本连杆。向量从表面投影点Q出发，黑点为候选P。N默认是微面法向，尚未认证接触区域；橙点为中指已有用户方向覆盖。棱边无唯一法向，切向不是滑移方向。所有坐标在当前远节连杆系，未应用腕/世界变换。'
        table='<table border="1"><tr><th>全局ID</th><th>local</th><th>方向来源</th><th>原表面偏移mm</th><th>N (local)</th></tr>'+''.join(f'<tr><td>{x["index"]}</td><td>{x["local_index"]}</td><td>{x["normal_source"]}</td><td>{x["distance_to_hand_surface_mm"]:.5f}</td><td>{html.escape(str(np.round(x["N"],5).tolist()))}</td></tr>' for x in frames)+'</table>'
        (a.output/f'{link}.html').write_text('<!doctype html><meta charset="utf-8"><a href="index.html">五指目录</a><p>'+caption+'</p>'+fig.to_html(full_html=False,include_plotlyjs=True)+table)
        (a.output/f'{link}_frames.json').write_text(json.dumps(frames,indent=2))
        summary.append(dict(link=link,count=len(ids),global_indices=ids,corrected_count=sum(x['normal_source']=='user_confirmed_override' for x in frames)))
        print(link,len(ids),'points',flush=True)
    (a.output/'manifest.json').write_text(json.dumps(dict(links=summary,urdf_sha256=sha(urdf),contacts_sha256=sha(urdf.parent/'contact_points.json'),override_sha256=sha(a.normal_overrides),vector_mm=a.vector_mm,scope='Local visual inspection only, no semantic/dynamic admission'),indent=2))
    links=''.join(f'<li><a href="{x["link"]}.html">{x["link"]}</a>：{x["count"]}点；用户方向覆盖{x["corrected_count"]}点</li>' for x in summary)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>L20五指远节局部核查</h1><p>每页只显示该远节原网格、连杆坐标轴、全部接触点和N/T1/T2。默认全部点仅显示N；下拉可切单点或全部切向。请反馈连杆名＋全局点号。</p><ul>'+links+'</ul>')
    print('Complete:',a.output.resolve()/'index.html')


if __name__=='__main__':main()
