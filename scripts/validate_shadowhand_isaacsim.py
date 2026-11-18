"""Validate generated closed ShadowHand poses in native Isaac Sim SDF, fresh process per direction."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
from prepare_shadowhand_isaacsim import prepare,ROOT

CASES=['gx+','gx-','gy+','gy-','gz+','gz-']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--object-mesh',type=Path,default=ROOT/'data/meshdata/power-drill/coacd/decomposed.obj')
    p.add_argument('--physical-metadata',type=Path,help='Explicit assumed mass and bbox inertia, in this mesh frame')
    p.add_argument('--prepared-run',action='store_true',help='Run exactly the already prepared inputs and plan; not trial resume')
    p.add_argument('--top-k',type=int,default=16);p.add_argument('--indices',type=int,nargs='+')
    p.add_argument('--cases',choices=CASES,nargs='+',default=CASES)
    p.add_argument('--gravity-seconds',type=float,default=10.);p.add_argument('--sdf-resolution',type=int,default=256)
    p.add_argument('--kp',type=float,default=5.);p.add_argument('--kd',type=float,default=.05)
    p.add_argument('--gui',action='store_true');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    import numpy as np
    if a.top_k<1 or not np.isfinite([a.gravity_seconds,a.kp,a.kd]).all() or a.gravity_seconds<1 or a.kp<=0 or a.kd<0 or a.sdf_resolution<32 or len(set(a.cases))!=len(a.cases):p.error('Invalid parameters')
    out=a.output.resolve()
    if out.exists() and not a.prepared_run:p.error('Output must be new')
    if not a.result_file.is_file():p.error('Generation NPY not found; wait for generation to finish')
    from prepare_shadowhand_isaacsim import sha
    if a.prepared_run:
        m=json.loads((out/'inputs/manifest.json').read_text())
        if m['source_sha256']!=sha(a.result_file.resolve()) or m['meshes'][-1]['sha256']!=sha(a.object_mesh.resolve()):p.error('Prepared source/mesh changed')
        if (out/'results.json').exists():p.error('Trial results already exist; prepared-run is not resume')
    else:
        m=prepare(a.result_file.resolve(),a.object_mesh.resolve(),out/'inputs',a.indices,a.top_k)
        if a.physical_metadata:
            import shutil
            import numpy as np
            meta=json.loads(a.physical_metadata.read_text());mass=float(meta['mass_kg_assumed']);diag=np.asarray(meta['diaginertia_kg_m2'],float)
            extent=np.asarray(m['object_extents_m']);expected=mass/12*np.array([sum(extent[[1,2]]**2),sum(extent[[0,2]]**2),sum(extent[[0,1]]**2)])
            if not np.isfinite(mass) or mass<=0 or diag.shape!=(3,) or not np.isfinite(diag).all() or not np.allclose(diag,expected,rtol=1e-5):p.error('Physical metadata mismatch with object dimensions')
            m.update(object_mass_kg=mass,object_com=[0.,0.,0.],object_inertia=np.diag(diag).tolist(),physical_metadata_sha256=sha(a.physical_metadata.resolve()),mass_is_measured=False,inertia_model=meta['inertia_model'])
            shutil.copy2(a.physical_metadata,out/'physical_metadata.json')
        raw=np.load(a.result_file,allow_pickle=True)
        for c in m['candidates']:
            g=raw[c['index']]
            c['generation_metrics']={n:g[n] for n in ['nearest_contact_class_ids','max_safe_contact_distance_m','min_excluded_surface_sample_distance_m','semantic_verified'] if n in g}
        (out/'inputs/manifest.json').write_text(json.dumps(m,indent=2))
    steps=round(a.gravity_seconds*240)
    plan=dict(protocol='direct',gravity_steps=steps,dt=1/240,kp=a.kp,kd=a.kd,effort_nm=1.,friction=3.,sdf_resolution=a.sdf_resolution,source_sha256=m['source_sha256'],cases=a.cases,candidate_indices=[c['index'] for c in m['candidates']],note='Static generated targets, no opening, contact adjustment, RRL or fabricated mimic. Fresh OS process per direction. Reuses audited native SDF backend.')
    plan.update(object_mesh_sha256=sha(a.object_mesh.resolve()),physical_metadata_sha256=sha(a.physical_metadata.resolve()) if a.physical_metadata else None)
    if a.prepared_run:
        if json.loads((out/'plan.json').read_text())!=plan:p.error('Prepared plan changed')
    else:(out/'plan.json').write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return
    from replay_l20_isaacsim_html import export
    rows=[]
    def report():
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        cells=[]
        for c in m['candidates']:
            rr=[r for r in rows if r['candidate']==c['index']]
            by={r['case']:r for r in rr};cols=[]
            for case in a.cases:
                r=by.get(case)
                label='待运行' if r is None else '执行异常' if not r['completed'] else '通过' if r['result']['passed'] else '失败'
                link=f"candidate_{c['index']:03d}/{case}/replay.html"
                cols.append(f'<td><a href="{link}">{label}</a></td>')
            cells.append(f'<tr><td>{c["rank"]}</td><td>{c["index"]}</td>'+''.join(cols)+'</tr>')
        complete=len(rows)==len(m['candidates'])*len(a.cases) and all(r['completed'] for r in rows)
        (out/'status.json').write_text(json.dumps(dict(complete=complete,completed_trials=sum(r['completed'] for r in rows),expected_trials=len(m['candidates'])*len(a.cases)),indent=2))
        text=f'# ShadowHand 电钻静态位姿验证\n\n完整：{complete}。逐方向结果见 results.json；固定原始关节目标，六方向重力10秒（默认）。\n\n无张开闭合、接触修正或RRL。手部沿用官方生成的简化几何转SDF；电钻为非凸实体SDF。力矩上限1Nm，默认Kp5/Kd0.05。\n\n判据：最大滑移≤5mm、转角≤15°、穿透≤1mm，末尾连续1秒有法向合力>0.001N；执行异常单独标注。\n'
        text+=f'\n物体质量：{m["object_mass_kg"]:.6f}kg；使用显式元数据时为假设值。碰撞源：{a.object_mesh.resolve()}。\n'
        text+='\n语义生成指标记录于inputs/manifest.json各候选generation_metrics；本轮不按语义指标删除候选。动力学通过不代表全手接触区域或闭合路径已合格。\n'
        (out/'REPORT.md').write_text(text)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>ShadowHand 电钻验证</title><h1>ShadowHand 电钻静态抓取验证</h1><p>'+html.escape(text)+'</p><table border="1"><tr><th>能量排名</th><th>原候选下标</th>'+''.join('<th>'+v+'</th>' for v in a.cases)+'</tr>'+''.join(cells)+'</table><p>链接为保存轨迹回放，不是新的物理实验。</p>')
    report()
    for c in m['candidates']:
        for case in a.cases:
            folder=out/f'candidate_{c["index"]:03d}'/case
            folder.parent.mkdir(parents=True,exist_ok=True)
            cmd=['bash',str(ROOT/'scripts/run_l20_isaacsim.sh'),'--inputs',str(out/'inputs'),'--output',str(folder),'--indices',str(c['index']),'--cases',case,'--protocol','direct','--steps',str(steps),'--kp',str(a.kp),'--kd',str(a.kd),'--sdf-resolution',str(a.sdf_resolution)]
            if a.gui:cmd+=['--gui','--viewer-seconds','3']
            print(f'ShadowHand rank {c["rank"]} candidate {c["index"]} {case}',flush=True)
            with folder.with_suffix('.log').open('w') as log:run=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(candidate=c['index'],case=case,completed=False,returncode=run.returncode)
            if run.returncode==0 and (folder/'summary.json').exists():
                data=json.loads((folder/'summary.json').read_text());row.update(completed=True,result=data['results'][0])
                export(folder,out/'inputs',folder/'replay.html')
                rp=folder/'REPORT.md';rp.write_text(rp.read_text().replace('L20','ShadowHand').replace('全部手部 STL','官方生成器手部碰撞几何'))
            rows.append(row);report()
    print(out/'index.html')
    if not all(r['completed'] for r in rows):raise SystemExit('Some processes failed; inspect per-direction .log and error.json')

if __name__=='__main__':main()
