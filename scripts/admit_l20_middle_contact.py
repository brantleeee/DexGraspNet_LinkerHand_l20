"""CPU region review and explicit C contact-candidate admission. No dynamics."""
import argparse,hashlib,html,json,sys
from pathlib import Path
import numpy as np
import trimesh
from robust_surface_query import nearest_surface
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'grasp_generation'))
DEFAULT=ROOT/'data/experiments/l20_contact_reselect_full_check_20261116'
PLAN=ROOT/'data/experiments/l20_clean_drill_sdf_20261115_225533/plan.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def binding_id(binding):
    return hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()


def verify_binding(meta):
    if binding_id(meta['binding'])!=meta['review_id']:raise ValueError('Review binding ID mismatch')
    for path,h in meta['binding']['files'].items():
        if sha(path)!=h:raise ValueError('Review source/model changed: '+path)


def decision(annotation,distance_mm,inside,admissible,threshold_mm=1.):
    if not admissible:return 'source_geometry_rejected'
    if annotation.get('reviewed') is not True:return 'region_not_reviewed'
    if annotation.get('label')!='palmar':return 'region_not_allowed'
    if inside:return 'inside_physics_proxy'
    if not np.isfinite(distance_mm) or distance_mm>threshold_mm:return 'outside_contact_band'
    return 'contact_candidate_admitted'


def solid_from_plan(path):
    plan=json.loads(path.read_text());solid=path.parent/'collision_closed.obj'
    original=ROOT/'data/meshdata/power-drill-clean-l20/coacd/decomposed.obj'
    if sha(solid)!=plan['collision_sha256'] or sha(original)!=plan['generation_mesh_sha256']:raise ValueError('Physics proxy/generation source mismatch')
    mesh=trimesh.load(solid,force='mesh',process=True)
    if not(mesh.is_volume and mesh.is_watertight and mesh.is_winding_consistent):raise ValueError('Need closed consistently oriented positive-volume proxy')
    return mesh,solid,original


def outside_query(mesh,points):
    if not len(points):return np.array([]),np.array([],dtype=bool)
    _,distance,_=nearest_surface(mesh,points)
    center=mesh.bounds.mean(0)
    normalized=trimesh.Trimesh((mesh.vertices-center)*1000,mesh.faces,process=False)
    inside=normalized.contains((np.asarray(points)-center)*1000)
    # A point within numerical surface tolerance is treated as on the boundary.
    inside=np.asarray(inside)&(distance>1e-9)
    return distance*1000,inside


def load_hand():
    from diagnose_l20_middle_objective import URDF
    from utils.hand_model import HandModel
    return HandModel(model_path=str(URDF),mesh_path=str(URDF.parent/'meshes'),contact_points_path=str(URDF.parent/'contact_points.json'),penetration_points_path=str(URDF.parent/'penetration_points.json'),device='cpu'),URDF


def prepare(a):
    overrides={}
    if a.normal_overrides:
        annotation=json.loads(a.normal_overrides.read_text())
        if annotation.get('schema')!='l20-user-contact-normal-v1' or annotation.get('coordinate_frame')!='middle_distal':raise ValueError('Unexpected contact-normal override frame')
        if sha(annotation['source_frames'])!=annotation['source_frames_sha256']:raise ValueError('Source frame for manual directions changed')
        reference={str(x['index']):x for x in json.loads(Path(annotation['source_frames']).read_text())}
        for idx,record in annotation['points'].items():
            rule=record['rule'];axis=rule.lstrip('-')
            if axis not in ['N','T1','T2'] or rule not in ['N','-N','T1','-T1','T2','-T2']:raise ValueError('Invalid frozen-frame direction rule')
            expected=np.asarray(reference[idx][axis])*(-1 if rule.startswith('-') else 1)
            if not np.allclose(record['N'],expected,atol=1e-12) or not np.allclose(record['reference_N'],reference[idx]['N'],atol=1e-12):raise ValueError('Manual direction does not match the frozen reference')
        for path,h in annotation['asset_hashes'].items():
            if sha(path)!=h:raise ValueError('Contact-normal override assets changed')
        overrides=annotation['points']
    source=a.run.resolve();provenance=json.loads((source/'config.json').read_text())
    if not json.loads((source/'status.json').read_text())['complete']:raise ValueError('Incomplete source enumeration')
    if provenance['diagnostics_only']:raise ValueError('Need full-grid enumeration, not four-pose-only run')
    grid=Path(provenance['source']);cfg=json.loads((grid/'config.json').read_text())
    for path,h in cfg['sha256'].items():
        if sha(path)!=h:raise ValueError('Source code/model changed: '+path)
    for name,h in provenance['source_files_sha256'].items():
        if sha(grid/name)!=h:raise ValueError('Source grid altered')
    mesh,solid,original=solid_from_plan(a.physics_plan.resolve())
    from utils.grasp_results import verify_mesh_identity
    if verify_mesh_identity(cfg['result_file'],original) is None:raise ValueError('Missing generation mesh identity')
    raw=np.load(cfg['result_file'],allow_pickle=True)
    if any(abs(float(raw[c]['scale'])-1)>1e-7 for c in cfg['candidates']):raise ValueError('Contact C requires saved scale1; no wrist rescaling')
    hand,urdf=load_hand();owners=[list(hand.mesh)[i] for i in hand.global_index_to_link_index.tolist()]
    if len(set(a.points))!=len(a.points) or any(i<0 or i>=len(owners) or not owners[i].startswith('middle_') for i in a.points):raise ValueError('Review unique middle-finger contact indices only')
    import xml.etree.ElementTree as ET
    files=[source/'config.json',source/'results.json',source/'selected_references.json',grid/'config.json',grid/'results.json',Path(cfg['result_file']),urdf,urdf.parent/'contact_points.json',solid,original,a.physics_plan.resolve()]
    files += [Path(path) for path in cfg['sha256']]
    for link in ET.parse(urdf).getroot().findall('link'):
        for m in link.findall('visual/geometry/mesh'):files.append((urdf.parent/m.get('filename')).resolve())
    files.append(ROOT/'scripts/robust_surface_query.py')
    files.append(ROOT/'scripts/contact_frame_display.py')
    if a.normal_overrides:files.append(a.normal_overrides.resolve())
    binding=dict(run=str(source),grid=str(grid),points=a.points,files={str(f.resolve()):sha(f) for f in files},physics_plan=str(a.physics_plan.resolve()),geometry_review_version='scale-normalized-v2')
    meta=dict(binding=binding,review_id=binding_id(binding),review_note='Self-reported anatomical review, not physics certification',implementation_sha256=sha(Path(__file__)))
    a.output.mkdir(parents=True);save(a.output/'review_manifest.json',meta)
    data=json.loads((source/'results.json').read_text());preview=[]
    for d in data:
        if d['row_index'] not in [20,43]:continue
        candidates={x['index']:x for x in d['middle_candidates']};pts=np.array([candidates[i]['point_m'] for i in a.points])
        distances,inside=outside_query(mesh,pts)
        for i,dist,ins in zip(a.points,distances,inside):preview.append(dict(row_index=d['row_index'],candidate=d['candidate'],point=i,proxy_distance_mm=float(dist),proxy_inside=bool(ins),geometry_admissible=d['geometry_admissible'],region='unknown'))
    save(a.output/'outside_preview.json',preview)
    import plotly.graph_objects as go
    from contact_frame_display import contact_basis,add_arrow,add_link_axes
    plots=[];point_info=[];normal_info=[];patch_plots=[];frames=[]
    for link in sorted({owners[i] for i in a.points}):
        m=hand.mesh[link];v=m['vertices'].numpy();f=m['faces'].numpy()
        fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='lightgray',opacity=.85,name=link,flatshading=True))
        add_link_axes(fig,link)
        ids=[i for i in a.points if owners[i]==link];pts=hand.contact_candidates[ids].numpy()
        tm=trimesh.Trimesh(v,f,process=False);closest,dist,face=nearest_surface(tm,pts)
        for idx,pt,cp,gap,fi in zip(ids,pts,closest,dist,face):
            from contact_normal_debug import normal_witness
            witness=normal_witness(tm,pt,cp,fi);witness['index']=idx;normal_info.append(witness)
            normal=np.array(witness['normal'])
            group=f'point{idx}'
            fig.add_trace(go.Scatter3d(x=[pt[0]],y=[pt[1]],z=[pt[2]],text=[f'P{idx}'],mode='markers+text',marker=dict(size=6),name=f'点{idx}：N/T1/T2',legendgroup=group))
            n,t1,t2=contact_basis(normal,tm.triangles[fi,1]-tm.triangles[fi,0])
            normal_source='selected_mesh_facet'
            if str(idx) in overrides:
                from contact_frame_display import override_basis
                n,t1,t2=override_basis(overrides[str(idx)])
                normal_source='user_confirmed_contact_direction'
                old_tip=cp+.002*normal
                fig.add_trace(go.Scatter3d(x=[cp[0],old_tip[0]],y=[cp[1],old_tip[1]],z=[cp[2],old_tip[2]],mode='lines',line=dict(color='gray',dash='dash'),name=f'{idx} 原微面N_facet',visible='legendonly'))
            for direction,color,label in [(n,'red','N'),(t1,'green','T1'),(t2,'blue','T2')]:
                add_arrow(fig,cp,direction,.002,color,f'{idx} {label}',group)
            fig.add_trace(go.Scatter3d(x=[pt[0],cp[0]],y=[pt[1],cp[1]],z=[pt[2],cp[2]],mode='markers+lines',marker=dict(size=3,color='black'),line=dict(color='black',dash='dash'),name=f'{idx} P→Q',legendgroup=group,showlegend=False))
            frames.append(dict(index=idx,coordinate_frame=link,link_frame_origin_m=[0.,0.,0.],candidate_P_m=pt.tolist(),surface_anchor_Q_m=cp.tolist(),N=n.tolist(),T1=t1.tolist(),T2=t2.tolist(),geometric_N=normal.tolist(),normal_source=normal_source,override_rule=overrides.get(str(idx),{}).get('rule'),tangent_definition='T1 from reference facet N for user override, otherwise triangle edge; T2=N cross T1; not motion directions',facet_normal_ambiguous_at_edge=witness['projection_on_edge']))
            patch=go.Figure()
            for triangle,color,name in [(witness['triangle'],'red','所取三角面')]+[(n['triangle'],'cyan','相邻面') for n in witness['adjacent_faces']]:
                tri=np.asarray(triangle);patch.add_trace(go.Mesh3d(x=tri[:,0],y=tri[:,1],z=tri[:,2],i=[0],j=[1],k=[2],color=color,opacity=.7,name=name,flatshading=True))
            tangent=tm.triangles[fi,1]-tm.triangles[fi,0];tangent/=np.linalg.norm(tangent)
            for direction,color,name,length in [(normal,'red','该三角面法向',.001),(tangent,'green','该三角面切向',.001)]:
                end=cp+length*direction
                patch.add_trace(go.Scatter3d(x=[cp[0],end[0]],y=[cp[1],end[1]],z=[cp[2],end[2]],mode='lines',line=dict(color=color,width=7),name=name))
                patch.add_trace(go.Cone(x=[end[0]],y=[end[1]],z=[end[2]],u=[direction[0]],v=[direction[1]],w=[direction[2]],sizemode='absolute',sizeref=.0001,anchor='tip',showscale=False,showlegend=False,colorscale=[[0,color],[1,color]]))
            patch.add_trace(go.Scatter3d(x=[pt[0],cp[0]],y=[pt[1],cp[1]],z=[pt[2],cp[2]],mode='markers+lines',name='候选点→最近表面点（非面法向）',line=dict(color='black',dash='dash')))
            patch.update_layout(scene_aspectmode='data',scene_camera_projection_type='orthographic',title=f'点{idx}局部：红=所取面及法向；绿=切向；青=相邻面；边缘法向不唯一')
            patch_plots.append(patch.to_html(full_html=False,include_plotlyjs=False))
            point_info.append(dict(index=idx,link=link,local_point_m=pt.tolist(),distance_to_hand_surface_mm=float(gap*1000),projected_point_m=cp.tolist(),hand_face_normal=normal.tolist()))
        buttons=[dict(label=label,method='relayout',args=[{'scene.camera.eye':eye}]) for label,eye in [('＋X',dict(x=2,y=0,z=.2)),('−X',dict(x=-2,y=0,z=.2)),('＋Y',dict(x=0,y=2,z=.2)),('−Y',dict(x=0,y=-2,z=.2)),('＋Z',dict(x=.1,y=0,z=2)),('−Z',dict(x=.1,y=0,z=-2))]]
        fig.update_layout(scene_aspectmode='data',scene=dict(xaxis_title='X_link (m)',yaxis_title='Y_link (m)',zaxis_title='Z_link (m)'),title=link+' 坐标系：N红 / T1绿 / T2蓝；向量起点Q为投影点，P为候选点',legend=dict(groupclick='togglegroup'),updatemenus=[dict(type='buttons',buttons=buttons)])
        plots.append(fig.to_html(full_html=False,include_plotlyjs=True if not plots else False))
    save(a.output/'point_info.json',point_info)
    save(a.output/'normal_diagnostics.json',normal_info)
    save(a.output/'contact_frames.json',frames)
    template=dict(schema='l20-contact-region-v1',review_id=meta['review_id'],reviewer='',reviewed_at='',labels={str(i):dict(label='unknown',reviewed=False,note='') for i in a.points})
    save(a.output/'labels.template.json',template)
    form='<p>核查者：<input id="reviewer"></p>'
    for i in a.points:
        form+=f'<p>点{i} <select id="label{i}"><option value="unknown">未知</option><option value="palmar">掌侧/指腹允许区</option><option value="dorsal">指背</option><option value="joint_shell">关节外壳</option></select> <label><input type="checkbox" id="checked{i}">已核查该点</label> 依据：<input id="note{i}"></p>'
    js='''<script>const payload=PAYLOAD; function download(){payload.reviewer=document.getElementById('reviewer').value.trim(); if(!payload.reviewer){alert('请填写核查者');return;} for(const id of Object.keys(payload.labels)){const e=payload.labels[id];e.label=document.getElementById('label'+id).value;e.reviewed=document.getElementById('checked'+id).checked;e.note=document.getElementById('note'+id).value.trim();if(e.reviewed&&!e.note){alert('已核查点请填写依据');return;}}payload.reviewed_at=new Date().toISOString();const u=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='l20_contact_labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}</script>'''.replace('PAYLOAD',json.dumps(template).replace('<','\\u003c'))
    text='原网格区域核查：只标注能确认的指腹/计划夹持面。未知不放行。'+('主图N采用用户确认的接触方向覆盖；灰虚线原微面N_facet可在图例打开。下面局部证据图仍显示原网格面法向，与主图有效接触方向分开。' if overrides else '主图N是最近单个三角面的法向，棱边不代表唯一接触法向。')+'局部坐标不自动定义掌侧。覆盖不会修改STL、生成器物体法向或E_fc。下载JSON后运行apply；该页面不会提交或启动仿真。'
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>中指区域核查</h1><p>'+text+'</p>'+''.join(plots)+'<h2>法向与切向局部证据</h2>'+''.join(patch_plots)+'<pre>'+html.escape(json.dumps(point_info,indent=2,ensure_ascii=False))+'</pre>'+form+'<button onclick="download()">下载标注JSON</button>'+js+'<h2>同源闭合代理的逐构型外侧预览</h2><pre>'+html.escape(json.dumps(preview,indent=2,ensure_ascii=False))+'</pre>')
    print('Prepared:',a.output.resolve()/'index.html')


def apply(a):
    review=a.review.resolve();meta=json.loads((review/'review_manifest.json').read_text());verify_binding(meta)
    if meta['binding'].get('geometry_review_version')!='scale-normalized-v2':raise ValueError('Old projection review; regenerate v2 page and review again')
    labels=json.loads(a.labels.read_text())
    if labels.get('schema')!='l20-contact-region-v1' or labels.get('review_id')!=meta['review_id']:raise ValueError('Labels do not match this review')
    if not labels.get('reviewer','').strip() or not labels.get('reviewed_at'):raise ValueError('Need reviewer identity and review timestamp')
    if set(labels['labels'])!={str(i) for i in meta['binding']['points']}:raise ValueError('Unexpected point annotations')
    for annotation in labels['labels'].values():
        if annotation.get('label') not in ['unknown','palmar','dorsal','joint_shell']:raise ValueError('Invalid annotation')
        if annotation.get('reviewed') is True and not annotation.get('note','').strip():raise ValueError('Reviewed point needs rationale')
    mesh,solid,_=solid_from_plan(Path(meta['binding']['physics_plan']))
    source=Path(meta['binding']['run']);data=json.loads((source/'results.json').read_text());cfg=json.loads((source/'config.json').read_text());grid=Path(cfg['source'])
    gridcfg=json.loads((grid/'config.json').read_text());gridrows=json.loads((grid/'results.json').read_text());raw=np.load(gridcfg['result_file'],allow_pickle=True)
    # Only reviewed allowed points require expensive geometric queries. Never prefilter by the old sign.
    queries=[];items=[]
    for d in data:
        points={x['index']:x for x in d['middle_candidates']}
        for e in d['enumeration']:
            annotation=labels['labels'].get(str(e['middle_contact']),{})
            item=dict(row_index=d['row_index'],candidate=d['candidate'],middle_contact=e['middle_contact'],total=e['total'],source_geometry_admissible=e['admissible'],region_label=annotation.get('label','unknown'),reason=decision(annotation,np.inf,False,e['admissible'],a.threshold_mm),proxy_distance_mm=None,proxy_inside=None)
            if e['admissible'] and annotation.get('reviewed') is True and annotation.get('label')=='palmar':
                queries.append(points[e['middle_contact']]['point_m']);item['query_index']=len(queries)-1
            items.append(item)
    distances,inside=outside_query(mesh,np.array(queries)) if queries else ([],[])
    for item in items:
        if 'query_index' not in item:continue
        j=item.pop('query_index');dist=float(distances[j]);ins=bool(inside[j]);annotation=labels['labels'][str(item['middle_contact'])]
        item.update(proxy_distance_mm=dist,proxy_inside=ins,reason=decision(annotation,dist,ins,True,a.threshold_mm))
    selected=[]
    hand=None
    for candidate in sorted({d['candidate'] for d in data}):
        eligible=[r for r in items if r['candidate']==candidate and r['reason']=='contact_candidate_admitted']
        if not eligible:selected.append(dict(candidate=candidate,status='no_valid_middle_contact',reference=None,dynamics_admitted=False));continue
        best=min(eligible,key=lambda x:x['total']);row=gridrows[best['row_index']]
        if hand is None:hand,_=load_hand()
        from diagnose_l20_middle_objective import pose_from_record
        pose,_=pose_from_record(raw[candidate],hand,'cpu');pose[0,9+hand.joints_names.index('middle_mcp_pitch')]=row['mcp_rad'];pose[0,9+hand.joints_names.index('middle_pip')]=row['pip_rad'];pose=hand.joint_constraints.project(pose)
        q=dict(raw[candidate]['qpos']);q.update({n:float(pose[0,9+i]) for i,n in enumerate(hand.joints_names)})
        selected.append(dict(**best,status='contact_candidate_admitted',reference=dict(qpos=q,scale=1.),static_geometry_admitted=False,execution_admitted=False,dynamics_admitted=False,pending=['absolute_hand_object_geometry','open_hand_mesh_containment','continuous_closure_execution','physical_contact_and_dynamics']))
    a.output.mkdir(parents=True);save(a.output/'config.json',dict(review=str(review),review_id=meta['review_id'],labels_sha256=sha(a.labels),threshold_mm=a.threshold_mm,physics_proxy_sha256=sha(solid),implementation_sha256=sha(Path(__file__))))
    save(a.output/'labels.json',labels);save(a.output/'admission_details.json',items);save(a.output/'selected_references.json',selected)
    if hand is not None:
        import plotly.graph_objects as go
        for s in selected:
            if s['reference'] is None:continue
            pose,_=pose_from_record(s['reference'],hand,'cpu');contacts=list(gridcfg['contact_ids']);contacts[cfg['middle_slot']]=s['middle_contact']
            import torch
            hand.set_parameters(pose,torch.tensor([contacts]))
            v=mesh.vertices;f=mesh.faces
            fig=go.Figure(hand.get_plotly_data(0,opacity=.6,with_contact_points=True)+[go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='orange',opacity=.4)])
            fig.update_layout(scene_aspectmode='data',title=f'C candidate {s["candidate"]}: contact-candidate only; dynamics blocked')
            fig.write_html(a.output/f'{s["candidate"]}_C.html',include_plotlyjs=True)
    lines=['# C有效中指接触准入','', '仅区域自报核查+同源闭合代理外侧接近+原几何可行域。不是绝对全手几何/路径/动力学认证。没有合格组合时不回退A/B。','', '|候选|状态|构型行|点号|代理外侧距离mm|','|---|---|---|---|---|']
    for s in selected:lines.append(f'|{s["candidate"]}|{s["status"]}|{s.get("row_index","—")}|{s.get("middle_contact","—")}|{s.get("proxy_distance_mm","—")}|')
    text='\n'.join(lines);(a.output/'REPORT.md').write_text(text)
    links=''.join(f'<p><a href="{s["candidate"]}_C.html">候选{s["candidate"]} C</a></p>' for s in selected if s['reference'] is not None)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><pre>'+html.escape(text)+'</pre>'+links)
    save(a.output/'status.json',dict(complete=True,contact_admitted=sum(s['reference'] is not None for s in selected),static_geometry_admitted=False,execution_admitted=False,dynamics_started=False))
    print('Complete:',a.output.resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    r=sub.add_parser('review');r.add_argument('--run',type=Path,default=DEFAULT);r.add_argument('--physics-plan',type=Path,default=PLAN);r.add_argument('--points',nargs='+',type=int,default=[211,219,228,231]);r.add_argument('--normal-overrides',type=Path);r.add_argument('--output',required=True,type=Path)
    c=sub.add_parser('apply');c.add_argument('--review',required=True,type=Path);c.add_argument('--labels',required=True,type=Path);c.add_argument('--threshold-mm',type=float,default=1.);c.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    if a.mode=='apply' and (not np.isfinite(a.threshold_mm) or a.threshold_mm<=0):p.error('Invalid threshold')
    prepare(a) if a.mode=='review' else apply(a)


if __name__=='__main__':main()
