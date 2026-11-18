"""Official DexGraspNet hand points: analytic capsule N vs runtime mesh facets."""
import argparse,hashlib,html,json,os,sys
from pathlib import Path
import numpy as np
import trimesh
from contact_frame_display import contact_basis,add_arrow,add_link_axes
from robust_surface_query import nearest_surface
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external/DexGraspNet_official/grasp_generation'


def capsule_surface(points,radius,height):
    points=np.asarray(points,dtype=float);axis=points.copy();axis[:,:2]=0;axis[:,2]=np.clip(axis[:,2],0,height)
    delta=points-axis;length=np.linalg.norm(delta,axis=1)
    if np.any(length<1e-12):raise ValueError('Capsule axis point has no unique outward normal')
    normal=delta/length[:,None]
    return axis+radius*normal,normal,length-radius


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--distal-only',action='store_true');a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    sys.path.insert(0,str(SOURCE))
    from utils.hand_model import HandModel
    import utils.hand_model as module
    if Path(module.__file__).resolve()!=SOURCE/'utils/hand_model.py':raise ValueError('Wrong HandModel imported')
    cwd=Path.cwd()
    try:
        os.chdir(SOURCE)
        hand=HandModel(mjcf_path='mjcf/shadow_hand_wrist_free.xml',mesh_path='mjcf/meshes',contact_points_path='mjcf/contact_points.json',penetration_points_path='mjcf/penetration_points.json',device='cpu')
    finally:os.chdir(cwd)
    import plotly.graph_objects as go
    a.output.mkdir(parents=True);summary=[];offset=0
    for link,m in hand.mesh.items():
        points=m['contact_candidates'].numpy();ids=list(range(offset,offset+len(points)));offset+=len(points)
        if not len(points) or a.distal_only and 'distal' not in link:continue
        if 'geom_param' not in m:raise ValueError('Contact-bearing noncapsule needs separate surface audit: '+link)
        radius=float(m['geom_param'][0]);height=float(m['geom_param'][1])*2
        q,n,d=capsule_surface(points,radius,height)
        v=m['vertices'].numpy();f=m['faces'].numpy();runtime=trimesh.Trimesh(v,f,process=False)
        qm,dm,fi=nearest_surface(runtime,points)
        # Place a display tessellation at the segment used by official cal_distance.
        capsule=trimesh.creation.capsule(radius=radius,height=height)
        capsule.apply_translation(np.array([0,0,height/2])-capsule.bounds.mean(0))
        safe=link.replace('robot0:','');fig=go.Figure();cv=capsule.vertices;cf=capsule.faces
        fig.add_trace(go.Mesh3d(x=cv[:,0],y=cv[:,1],z=cv[:,2],i=cf[:,0],j=cf[:,1],k=cf[:,2],color='lightgray',opacity=.4,name='官方解析胶囊的显示网格'))
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='orange',opacity=.4,name='当前HandModel运行库生成的网格',visible='legendonly'))
        add_link_axes(fig,link);base=[i for i in range(len(fig.data)) if i!=1];markers=[];normals=[];groups={};records=[]
        for j,idx in enumerate(ids):
            group=f'point{idx}';start=len(fig.data);markers.append(start)
            fig.add_trace(go.Scatter3d(x=[points[j,0]],y=[points[j,1]],z=[points[j,2]],mode='markers+text',text=[str(idx)],marker=dict(size=5,color='black'),name=f'{idx} / local {j}',legendgroup=group))
            nn,t1,t2=contact_basis(n[j],[0,0,1])
            for vec,color,label in [(nn,'red','N_capsule'),(t1,'green','T1'),(t2,'blue','T2')]:
                before=len(fig.data);add_arrow(fig,q[j],vec,.003,color,f'{idx} {label}',group)
                if label=='N_capsule':normals+=list(range(before,len(fig.data)))
            raw=runtime.face_normals[fi[j]]
            groups[idx]=list(range(start,len(fig.data)))
            angle=float(np.degrees(np.arccos(np.clip(raw@nn,-1,1))))
            records.append(dict(index=idx,local_index=j,coordinate_frame=link,P_m=points[j].tolist(),Q_capsule_m=q[j].tolist(),N_capsule=nn.tolist(),T1=t1.tolist(),T2=t2.tolist(),signed_outside_distance_capsule_mm=float(d[j]*1000),official_positive_inside_sdf_mm=float(-d[j]*1000),Q_runtime_mesh_m=qm[j].tolist(),runtime_mesh_distance_mm=float(dm[j]*1000),runtime_facet_normal=raw.tolist(),analytic_facet_angle_deg=angle,normal_origin='Derived from official analytic hand capsule; not stored in official contact_points.json and not E_fc object-query normal'))
        def visible(indices):return [i in indices for i in range(len(fig.data))]
        buttons=[dict(label='全部点：仅解析N',method='update',args=[{'visible':visible(base+markers+normals)}]),dict(label='全部N/T1/T2',method='update',args=[{'visible':[i!=1 for i in range(len(fig.data))]}])]
        buttons += [dict(label=f'仅点{idx}',method='update',args=[{'visible':visible(base+groups[idx])}]) for idx in ids]
        for i,t in enumerate(fig.data):t.visible=i in base+markers+normals and i!=1
        fig.update_layout(scene_aspectmode='data',height=900,title=f'官方 {link}: N红 / T1绿 / T2蓝，当前child局部坐标',updatemenus=[dict(buttons=buttons,x=0,y=1.12)],legend=dict(groupclick='togglegroup'))
        bounds_expected=np.array([[-radius,-radius,-radius],[radius,radius,height+radius]])
        mismatch=float(abs(runtime.bounds-bounds_expected).max())
        text='官方点文件仅含坐标；红N为官方手部解析胶囊的外法向，正内SDF梯度为−N。E_fc实际使用物体距离查询法向，不是这里的手部N。橙色运行库网格可在图例开关；若与解析模型偏移，不能据它直接判红N错误。'
        (a.output/f'{safe}.html').write_text('<!doctype html><meta charset="utf-8"><a href="index.html">目录</a><p>'+text+'</p>'+fig.to_html(full_html=False,include_plotlyjs=True)+'<pre>'+html.escape(json.dumps(records,ensure_ascii=False,indent=2))+'</pre>')
        (a.output/f'{safe}.json').write_text(json.dumps(records,indent=2))
        summary.append(dict(link=link,page=safe+'.html',points=len(ids),indices=ids,radius_m=radius,segment_height_m=height,analytic_expected_bounds_m=bounds_expected.tolist(),runtime_mesh_bounds_m=runtime.bounds.tolist(),max_bounds_difference_mm=mismatch*1000,max_abs_capsule_point_offset_mm=float(abs(d).max()*1000),median_normal_angle_deg=float(np.median([r['analytic_facet_angle_deg'] for r in records]))))
        print(link,len(ids),'points',flush=True)
    files=[SOURCE/'mjcf/contact_points.json',SOURCE/'mjcf/shadow_hand_wrist_free.xml',SOURCE/'utils/hand_model.py',SOURCE/'utils/object_model.py',SOURCE/'utils/energy.py']
    manifest=dict(source=str(SOURCE),HandModel_module=str(Path(module.__file__).resolve()),trimesh_version=trimesh.__version__,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},links=summary,stored_normals=False,scope='Local analytic hand-surface normal inspection, no object/grasp dynamics')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    lines=['# 官方ShadowHand接触点法向核查','', '官方JSON只存点坐标；这里推导手部解析胶囊外法向。E_fc使用物体查询法向，不能用本图认证E_fc正确。','', '当前Trimesh生成的HandModel显示胶囊与官方cal_distance解析段存在Z范围差异，保留橙色网格对照，未修改官方源。不能将运行库几何差异归为官方预存法向错误。','', '|连杆|点数|最大解析表面偏移mm|显示/解析bounds最大差mm|','|---|---:|---:|---:|']
    for s in summary:lines.append(f'|{s["link"]}|{s["points"]}|{s["max_abs_capsule_point_offset_mm"]:.5f}|{s["max_bounds_difference_mm"]:.5f}|')
    text='\n'.join(lines);(a.output/'REPORT.md').write_text(text)
    nav=''.join(f'<li><a href="{s["page"]}">{s["link"]}</a>：{s["points"]}点</li>' for s in summary)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>官方ShadowHand局部接触点核查</h1><ul>'+nav+'</ul><pre>'+html.escape(text)+'</pre>')
    print('Complete:',a.output.resolve()/'index.html')


if __name__=='__main__':main()
