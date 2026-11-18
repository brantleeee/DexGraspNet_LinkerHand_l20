"""Run each gravity direction in a fresh Isaac Sim process and merge audited output."""
import json
from pathlib import Path
import shutil
import subprocess

CASES=['gx+','gx-','gy+','gy-','gz+','gz-']


def run_isolated(command, output):
    output=Path(output)
    output.mkdir(parents=True)
    def save(name, value):
        (output/name).write_text(json.dumps(value,indent=2))
    save('status.json',dict(status='running',case_isolation='fresh_process'))
    summaries=[];failures=[]
    for case in CASES:
        folder=output/'directions'/case
        cmd=command.copy();cmd[cmd.index('--output')+1]=str(folder)
        cmd+=['--cases',case]
        with (output/f'{case}.log').open('w') as log:
            run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        state=folder/'status.json';summary=folder/'summary.json'
        if run.returncode or not state.exists() or json.loads(state.read_text()).get('status')!='complete' or not summary.exists():
            error=folder/'error.json'
            failures.append(dict(case=case,error=json.loads(error.read_text()) if error.exists() else dict(message=f'Process exited {run.returncode}')))
            continue
        data=json.loads(summary.read_text())
        if len(data['results'])!=1 or data['results'][0]['case']!=case:
            raise ValueError(f'Invalid isolated direction results: {folder}')
        stats=json.loads((folder/'runtime_collision_statistics.json').read_text())
        audit=json.loads((folder/'collision_audit.json').read_text())
        drives=json.loads((folder/'runtime_drives.json').read_text())
        kp=[v for v in drives['kp'][0] if v!=0];kd=[v for v in drives['kd'][0] if v!=0]
        if (not audit['all_sdf'] or not stats['available'] or stats['statistics']['nb_convex_shapes']!=0
                or stats['statistics']['nb_trimesh_shapes']!=23 or len(kp)!=16 or len(kd)!=16
                or any(abs(v-data['parameters']['kp'])>1e-5 for v in kp)
                or any(abs(v-data['parameters']['kd'])>1e-5 for v in kd)
                or not data['results'][0]['numerical_valid']):
            raise ValueError(f'Isolated direction audit failed: {folder}')
        if summaries:
            first=summaries[0]
            params=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output','cases']}
            if data['manifest']!=first['manifest'] or params(data)!=params(first):
                raise ValueError(f'Isolated direction input/protocol mismatch: {folder}')
        summaries.append(data)
    save('case_isolation.json',dict(mode='fresh_process',cases=CASES,failures=failures))
    if failures:
        save('error.json',dict(type='IsolatedCaseFailure',message='One or more fresh-process directions failed',failures=failures))
        save('status.json',dict(status='failed',case_isolation='fresh_process'))
        return subprocess.CompletedProcess(command,1)
    first=output/'directions'/CASES[0]
    for name in ['scene.usda','protocol.json','collision_audit.json','runtime_collision_statistics.json','runtime_drives.json']:
        shutil.copy2(first/name,output/name)
    results=[data['results'][0] for data in summaries]
    for case in CASES:
        shutil.copy2(output/'directions'/case/f'trajectory_{case}.npz',output/f'trajectory_{case}.npz')
    summary=summaries[0].copy()
    summary['parameters']=dict(summary['parameters'],output=str(output),cases=CASES)
    summary.update(results=results,all_directions_pass=all(r['passed'] for r in results),case_isolation='fresh_process')
    save('summary.json',summary);save('results.json',results)
    (output/'REPORT.md').write_text('# L20 独立进程六方向验证\n\n'
        '每方向使用全新 Isaac Sim 进程，避免跨方向复用求解器状态。碰撞、物理参数和验收阈值保持原设置。\n\n'
        '|方向|通过|详细报告|\n|---|---|---|\n'+''.join(
            f"|{r['case']}|{r['passed']}|[报告](directions/{r['case']}/REPORT.md)|\n" for r in results))
    save('status.json',dict(status='complete',all_directions_pass=summary['all_directions_pass']))
    return subprocess.CompletedProcess(command,0)
