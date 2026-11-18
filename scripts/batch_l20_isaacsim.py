"""Validate all saved L20 grasps at one scale in separate Isaac Sim processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime
from l20_object_catalog import OBJECTS
from l20_isolated_cases import run_isolated

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',choices=OBJECTS,required=True)
    p.add_argument('--scale',type=float,default=.06)
    p.add_argument('--indices',type=int,nargs='+')
    p.add_argument('--result-file',type=Path)
    p.add_argument('--object-mesh',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Audit and keep completed candidates; archive and retry incomplete candidates')
    p.add_argument('--isolate-cases',action='store_true',help='Run each direction of pending candidates in a fresh process')
    p.add_argument('--steps',type=int,default=2400,help='Gravity-on steps; default 10 seconds')
    p.add_argument('--protocol',choices=['acquire','direct'],default='acquire')
    p.add_argument('--close-seconds',type=float,default=2.)
    p.add_argument('--settle-seconds',type=float,default=1.)
    p.add_argument('--open-fraction',type=float,default=.6)
    p.add_argument('--sdf-resolution',type=int,default=256)
    p.add_argument('--reuse-results',type=Path,nargs='+',default=[],help='Reuse completed single-candidate results only after checking inputs, protocol and runtime collision/drive audits')
    a=p.parse_args()
    if a.output.exists() and not a.resume:p.error('Output must be new (or use --resume)')
    if a.resume and not (a.output/'inputs/manifest.json').is_file():p.error('Resume requires existing prepared inputs')
    a.output.mkdir(parents=True,exist_ok=a.resume)
    inputs=a.output/'inputs'
    cmd=[sys.executable,str(ROOT/'scripts/prepare_l20_isaacsim.py'),'--object',a.object,
         '--scale',str(a.scale),'--output',str(inputs)]
    for name in ['result_file','object_mesh']:
        if getattr(a,name):cmd+=['--'+name.replace('_','-'),str(getattr(a,name))]
    if a.indices is not None:cmd+=['--indices',*map(str,a.indices)]
    if a.resume:
        # Rebuild on CPU to verify current source poses, URDF and mesh hashes.
        with tempfile.TemporaryDirectory(prefix='l20_resume_') as tmp:
            fresh=Path(tmp)/'inputs'
            check=cmd.copy();check[check.index('--output')+1]=str(fresh)
            subprocess.run(check,check=True)
            if json.loads((fresh/'manifest.json').read_text())!=json.loads((inputs/'manifest.json').read_text()):
                p.error('Resume inputs changed; use a new output directory')
        for c in json.loads((inputs/'manifest.json').read_text())['candidates']:
            folder=a.output/f"candidate_{c['index']:03d}"
            state=folder/'status.json'
            if state.exists() and json.loads(state.read_text()).get('status')=='complete':
                a.reuse_results.append(folder)
    else:
        subprocess.run(cmd,cwd=ROOT,check=True)
    manifest=json.loads((inputs/'manifest.json').read_text())
    reusable={}
    for folder in a.reuse_results:
        data=json.loads((folder/'summary.json').read_text())
        old=data['manifest'];params=data['parameters']
        if params.get('protocol','direct')!=a.protocol:p.error(f'Cannot reuse a different acquisition protocol: {folder}')
        if a.protocol=='acquire' and any(params.get(k)!=getattr(a,k) for k in ['close_seconds','settle_seconds','open_fraction']):p.error(f'Reuse acquisition mismatch: {folder}')
        for key in ['object','scale','source_sha256','urdf_sha256','meshes','object_mass_kg','object_inertia']:
            if old[key]!=manifest[key]:p.error(f'Reuse input mismatch ({key}): {folder}')
        expected=dict(steps=a.steps,sdf_resolution=a.sdf_resolution,dt=1/240,kp=5.,kd=.05,
                      friction=3.,max_slip=.005,max_angle_deg=15.,max_penetration=.001)
        if any(params.get(k)!=v for k,v in expected.items()):p.error(f'Reuse protocol mismatch: {folder}')
        directions={'gx+','gx-','gy+','gy-','gz+','gz-'}
        if len(data['results'])!=6 or {r['case'] for r in data['results']}!=directions:p.error(f'Incomplete reuse directions: {folder}')
        if any(r['steps']!=a.steps or not r['numerical_valid'] for r in data['results']):p.error(f'Invalid reuse physics: {folder}')
        if json.loads((folder/'status.json').read_text()).get('status')!='complete':p.error(f'Incomplete reuse status: {folder}')
        audit=json.loads((folder/'collision_audit.json').read_text())
        stats=json.loads((folder/'runtime_collision_statistics.json').read_text())
        if not audit['all_sdf'] or not stats['available'] or stats['statistics']['nb_convex_shapes']!=0 or stats['statistics']['nb_trimesh_shapes']!=23:p.error(f'Reuse collision mismatch: {folder}')
        gains=json.loads((folder/'runtime_drives.json').read_text())
        kp=[v for v in gains['kp'][0] if v!=0];kd=[v for v in gains['kd'][0] if v!=0]
        if len(kp)!=16 or len(kd)!=16 or any(abs(v-5)>1e-5 for v in kp) or any(abs(v-.05)>1e-5 for v in kd):p.error(f'Reuse gain mismatch: {folder}')
        index=data['results'][0]['candidate']
        candidate=next((c for c in manifest['candidates'] if c['index']==index),None)
        previous=next((c for c in old['candidates'] if c['index']==index),None)
        if candidate is None or candidate!=previous:p.error(f'Reuse candidate mismatch: {folder}')
        reusable[index]=(folder.resolve(),data)
    results=[]
    for c in manifest['candidates']:
        i=c['index'];out=a.output/f'candidate_{i:03d}'
        if i in reusable:
            folder,data=reusable[i]
            results.append(dict(candidate=i,completed=True,reused=True,output=str(folder),
                                all_directions_pass=data['all_directions_pass'],results=data['results']))
            (a.output/'summary.json').write_text(json.dumps(results,indent=2))
            print(f'Reused audited {a.object} candidate {i}: {folder}',flush=True)
            continue
        cmd=['bash',str(ROOT/'scripts/run_l20_isaacsim.sh'),'--inputs',str(inputs),
             '--indices',str(i),'--steps',str(a.steps),'--sdf-resolution',str(a.sdf_resolution),'--output',str(out),
             '--protocol',a.protocol,'--close-seconds',str(a.close_seconds),'--settle-seconds',str(a.settle_seconds),'--open-fraction',str(a.open_fraction)]
        if out.exists():
            archive=a.output/'failed_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            archive.mkdir(parents=True)
            shutil.move(str(out),str(archive/out.name))
            previous_log=a.output/f'candidate_{i:03d}.log'
            if previous_log.exists():shutil.move(str(previous_log),str(archive/previous_log.name))
        print(f'Validating {a.object} candidate {i}',flush=True)
        with (a.output/f'candidate_{i:03d}.log').open('w') as log:
            if a.isolate_cases:
                print('Each direction uses a fresh Isaac Sim process.',file=log,flush=True)
                run=run_isolated(cmd,out)
            else:
                run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        summary=out/'summary.json'
        result=dict(candidate=i,completed=run.returncode==0 and summary.exists(),output=str(out))
        if result['completed']:
            data=json.loads(summary.read_text());result.update(all_directions_pass=data['all_directions_pass'],results=data['results'])
        results.append(result)
        (a.output/'summary.json').write_text(json.dumps(results,indent=2))
    (a.output/'REPORT.md').write_text('# L20 Isaac Sim SDF 批量验证\n\n'
        +f'物体：{a.object}，scale={a.scale}；每方向 {a.steps} 步，dt=1/240 秒，SDF 分辨率 {a.sdf_resolution}。\n\n'
        +f'协议：{a.protocol}；零重力闭合 {a.close_seconds} 秒，零重力保持 {a.settle_seconds} 秒，张开比例 {a.open_fraction}（direct 时跳过获取）。\n\n'
        +'手部完整 STL、物体已有网格全部使用 SDF；物体网格来源及精度限制见 inputs/manifest.json。\n\n'
        +'|候选|执行完成|通过方向数|六方向全部通过|结果|\n|---|---|---:|---|---|\n'
        +''.join(f"|{r['candidate']}|{r['completed']}|{sum(t['passed'] for t in r.get('results',[]))}/6|{r.get('all_directions_pass',False)}|[报告]({Path(r['output']).resolve()}/REPORT.md)|\n" for r in results))
    if not all(r['completed'] for r in results):raise SystemExit('Some simulations failed; inspect candidate logs.')


if __name__=='__main__':main()
