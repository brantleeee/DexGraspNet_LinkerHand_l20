"""Audit ranked L20 clean-drill grasps and validate in native Isaac Sim SDF.

prepare-only is CPU-only. No source poses or previous experiments are overwritten.
"""
import argparse,copy,hashlib,html,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data/experiments/l20_clean_drill_20261115/results/scales/scale_1/top_16/ranking.json'
CASES=['gx+','gx-','gy+','gy-','gz+','gz-']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_ranking(path,k):
    rank=json.loads(path.read_text());source=Path(rank['source']).resolve()
    if rank['object_code']!='power-drill-clean-l20' or sha(source)!=rank['source_sha256']:raise ValueError('Ranking object/source identity mismatch')
    rows=rank['rows'][:k]
    if k<1 or len(rows)!=k:raise ValueError('Not enough ranked candidates')
    raw=np.load(source,allow_pickle=True);ids=[r['candidate_index'] for r in rows]
    if len(set(ids))!=len(ids) or any(i<0 or i>=len(raw) for i in ids):raise ValueError('Invalid/duplicate source indices')
    for r in rows:
        g=raw[r['candidate_index']]
        if not np.isclose(float(g['scale']),1.,rtol=0,atol=1e-7) or not np.isclose(g['scale'],r['scale']) or not np.isclose(g['energy'],r['energy'],rtol=1e-7):raise ValueError('Ranked energy or saved scale mismatch')
    return rank,source,raw,rows,ids


def prepare(a):
    from validate_clean_graspqp_drill import closed_proxy
    rank,source,raw,rows,ids=audit_ranking(a.ranking.resolve(),a.top_k)
    mesh=ROOT/'data/meshdata/power-drill-clean-l20/coacd/decomposed.obj'
    sys.path.insert(0,str(ROOT/'grasp_generation'))
    from utils.grasp_results import verify_mesh_identity
    if verify_mesh_identity(source,mesh) is None:raise ValueError('Missing generation mesh identity')
    surface=trimesh.load(mesh,force='mesh',process=False);solid,repair=closed_proxy(surface)
    out=a.output.resolve()
    if out.exists():raise ValueError('Output must be new')
    out.mkdir(parents=True);collision=out/'collision_closed.obj';solid.export(collision)
    shutil.copy2(mesh,out/'generation_surface.obj');shutil.copy2(a.ranking,out/'ranking.json');shutil.copy2(source,out/'source_grasps.npy')
    cmd=[sys.executable,str(ROOT/'scripts/prepare_l20_isaacsim.py'),'--object','power-drill-clean-l20','--result-file',str(source),'--object-mesh',str(collision),'--scale','1','--indices',*map(str,ids),'--output',str(out/'inputs')]
    subprocess.run(cmd,cwd=ROOT,check=True)
    file=out/'inputs/manifest.json';m=json.loads(file.read_text());extent=np.asarray(m['object_extents_m']);mass=.6
    m.update(robot='LinkerHand L20',object_mass_kg=mass,object_com=surface.bounds.mean(0).tolist(),object_inertia=np.diag(mass/12*np.array([sum(extent[[1,2]]**2),sum(extent[[0,2]]**2),sum(extent[[0,1]]**2)])).tolist(),mass_source='user_confirmed_600g',inertia_model='uniform object bounding box approximation',object_mesh_kind='closed_nonconvex_proxy_from_frozen_clean_surface')
    by_index={r['candidate_index']:r for r in rows}
    for c in m['candidates']:
        c['rank']=by_index[c['index']]['rank'];c['generation_metrics']=by_index[c['index']]
    for j in m['joints']:
        if j['type']!='fixed' and not j['mimic']:j['limit']['effort']=a.effort_limit
    if sum(bool(j['mimic']) for j in m['joints'])!=5:raise ValueError('Expected 5 physical mimic joints')
    if sum(j['type']!='fixed' and not j['mimic'] for j in m['joints'])!=16:raise ValueError('Expected 16 independent L20 drives')
    file.write_text(json.dumps(m,indent=2))
    plan=dict(ranking_sha256=sha(a.ranking),source=str(source),source_sha256=sha(source),input_manifest_sha256=sha(file),input_arrays_sha256=sha(out/'inputs/meshes.npz'),generation_mesh_sha256=sha(mesh),collision_sha256=sha(collision),selected_indices=ids,selected_geometry_pass_count=sum(r['geometry_pass'] for r in rows),model='current L20 URDF',mimic_count=5,independent_drives=16,mass_kg=.6,effort_limit_nm=a.effort_limit,protocol=a.protocol,kp=a.kp,kd=a.kd,close_seconds=2.,settle_seconds=1.,open_fraction=.6,gravity_seconds=10.,sdf_resolution=a.sdf_resolution,dt=1/240,friction=3.,clearance_and_semantics_certified=False,repair=repair,note='Source poses and scale unchanged. Static geometry pass is not dynamics pass. Closed collision proxy + user mass and approximate bbox inertia. Separate fresh process per direction.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2));print(f'Prepared {len(ids)} candidates, {len(ids)*6} directions; {plan["selected_geometry_pass_count"]} geometry-filter passes; mass=.6kg',flush=True)
    return m,plan


def verify_prepared(a):
    out=a.output.resolve();plan=json.loads((out/'plan.json').read_text());m=json.loads((out/'inputs/manifest.json').read_text())
    if (out/'results.json').exists():raise ValueError('Results exist; this entry is not resume, use a new output')
    for file,key in [(out/'inputs/manifest.json','input_manifest_sha256'),(out/'inputs/meshes.npz','input_arrays_sha256'),(out/'collision_closed.obj','collision_sha256'),(a.ranking,'ranking_sha256'),(Path(plan['source']),'source_sha256')]:
        if sha(file)!=plan[key]:raise ValueError(f'Prepared input changed: {key}')
    for key in ['protocol','kp','kd','effort_limit','sdf_resolution']:
        stored=plan['effort_limit_nm' if key=='effort_limit' else key]
        if getattr(a,key)!=stored:raise ValueError(f'Prepared parameter changed: {key}')
    if len(m['candidates'])!=a.top_k:raise ValueError('Prepared candidate count changed')
    return m,plan


def run(a,m,plan):
    from replay_l20_isaacsim_html import export
    out=a.output.resolve();rows=[]
    def report():
        (out/'results.json').write_text(json.dumps(rows,indent=2));completed=sum(r['completed'] for r in rows);passed=sum(r['completed'] and r['result']['passed'] for r in rows)
        status=dict(attempted=len(rows),expected=len(m['candidates'])*6,completed=completed,passed=passed,execution_errors=len(rows)-completed,complete=len(rows)==len(m['candidates'])*6 and completed==len(rows))
        (out/'status.json').write_text(json.dumps(status,indent=2));text=f'# L20 clean电钻 Isaac Sim SDF验证\n\n{len(rows)}/{status["expected"]}方向已尝试，{completed}完成、{passed}通过、{len(rows)-completed}执行异常。\n\n协议{a.protocol}，质量600g，16位置驱动/5物理mimic；Kp={a.kp}、Kd={a.kd}、力矩上限{a.effort_limit}Nm。\n\nacquire为零重力闭合2秒/稳定1秒，再重力10秒；direct为直接闭合位姿重力10秒。完整非凸闭合碰撞代理，无支撑或焊接对象。\n\n通过：位移≤5mm、转角≤15°、包括准备阶段的采样最大穿透≤1mm，末尾连续1秒有效接触且数值正常；acquire还要求开重力前存在接触。物理通过不等于原开放表面或闭合轨迹/语义已合格。\n'
        (out/'REPORT.md').write_text(text)
        table=[]
        for c in m['candidates']:
            rr={r['case']:r for r in rows if r['candidate']==c['index']};cells=[]
            for case in CASES:
                r=rr.get(case);label='待运行' if r is None else '执行异常' if not r['completed'] else '通过' if r['result']['passed'] else '失败'
                cells.append(f'<td><a href="candidate_{c["index"]:03d}/{case}/replay.html">{label}</a></td>')
            table.append(f'<tr><td>{c["rank"]}</td><td>{c["index"]}</td><td>{c["generation_metrics"]["geometry_pass"]}</td>'+''.join(cells)+'</tr>')
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>L20 clean电钻 SDF动力学验证</h1><pre>'+html.escape(text)+'</pre><table border="1"><tr><th>排名</th><th>原下标</th><th>静态几何筛选</th>'+''.join('<th>'+k+'</th>' for k in CASES)+'</tr>'+''.join(table)+'</table><p>链接为已保存轨迹回放，不重新仿真。</p>')
    report()
    for c in m['candidates']:
        for case in CASES:
            folder=out/f'candidate_{c["index"]:03d}'/case;folder.parent.mkdir(parents=True,exist_ok=True)
            cmd=['bash',str(ROOT/'scripts/run_l20_isaacsim.sh'),'--inputs',str(out/'inputs'),'--output',str(folder),'--indices',str(c['index']),'--cases',case,'--protocol',a.protocol,'--close-seconds','2','--settle-seconds','1','--open-fraction','.6','--steps','2400','--sdf-resolution',str(a.sdf_resolution),'--kp',str(a.kp),'--kd',str(a.kd)]
            if a.gui:cmd+=['--gui','--viewer-seconds','3']
            print(f'L20 rank {c["rank"]}, candidate {c["index"]}, {case}',flush=True)
            with folder.with_suffix('.log').open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            row=dict(candidate=c['index'],case=case,completed=False,returncode=r.returncode)
            if r.returncode==0 and (folder/'summary.json').exists():
                data=json.loads((folder/'summary.json').read_text());row.update(completed=True,result=data['results'][0]);export(folder,out/'inputs',folder/'replay.html')
            rows.append(row);report()
    if not all(r['completed'] for r in rows):raise SystemExit('Some trials have execution errors; see case logs/error.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ranking',type=Path,default=DEFAULT);p.add_argument('--top-k',type=int,default=16);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--protocol',choices=['acquire','direct'],default='acquire');p.add_argument('--kp',type=float,default=5.);p.add_argument('--kd',type=float,default=.05);p.add_argument('--effort-limit',type=float,default=1.);p.add_argument('--sdf-resolution',type=int,default=256);p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared-run',action='store_true');p.add_argument('--gui',action='store_true')
    a=p.parse_args()
    if a.top_k<1 or not np.isfinite([a.kp,a.kd,a.effort_limit]).all() or min(a.kp,a.effort_limit)<=0 or a.kd<0 or a.sdf_resolution<32:p.error('Invalid parameters')
    m,plan=verify_prepared(a) if a.prepared_run else prepare(a)
    if not a.prepare_only:run(a,m,plan)

if __name__=='__main__':main()
