"""Two fresh processes: frozen policy vs applied-offset hold in settle tail."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.isolated import compare_traces
from isaacsim_grasp.reference import digest
from isaacsim_grasp.contact_timing import summarize_contact_timing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',type=Path,required=True,help='Completed trusted paired group folder')
    p.add_argument('--hold-start-seconds',type=float,default=2.6)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();source=a.baseline_run.resolve();out=a.output.resolve()
    if out.exists():p.error('Output must be new')
    spec=json.loads((source/'policy_plan.json').read_text());v=json.loads((source/'reproducibility.json').read_text())
    if not (v['process_reproducible'] and v['policy_evaluated'] and v['zero_policy_prefix']['passed']):p.error('Source is not a trusted paired evaluation')
    if spec['controller']!='policy' or spec.get('variant'):p.error('Requires original policy reference')
    cfg=spec['config'];gravity=cfg['close_seconds']+cfg['settle_seconds'];start=a.hold_start_seconds
    if not gravity-cfg['preload_seconds']<start<gravity:p.error('Hold start must lie inside preload window')
    if abs(start*60-round(start*60))>1e-7:p.error('Hold start must align to 60Hz policy boundary')
    # Reuse original runner's checkpoint/hash compatibility checks without changing it.
    cmd=[sys.executable,str(ROOT/'scripts/run_isaacsim_residual_v5.py'),'--mode','plan',
        '--object',spec['reference']['object'],'--candidate',str(spec['reference']['candidate']),
        '--num-envs','6','--position-noise',str(cfg['position_noise']),
        '--checkpoint',spec['checkpoint'],'--reference-results',spec['reference']['source_results'],'--output',str(out)]
    fresh=json.loads(subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True).stdout)
    if fresh['reference']!=spec['reference'] or fresh['config']!=cfg:p.error('Frozen config/reference differs from source')
    input_files=[spec['checkpoint'],spec['reference']['scene'],str(source/'policy_plan.json'),
        str(ROOT/'isaacsim_grasp/preload_hold.py'),str(ROOT/'scripts/isaacsim_preload_hold_case.py')]
    inputs={f:digest(f) for f in input_files}
    plan=dict(source=str(source),hold_start_seconds=start,gravity_start_seconds=gravity,inputs=inputs,native_processes=2)
    if a.plan_only:print(json.dumps(plan,indent=2));return
    out.mkdir(parents=True);(out/'plan.json').write_text(json.dumps(plan,indent=2))
    results={};trajectories={};timing={}
    for name in ['baseline','hold']:
        if any(digest(f)!=h for f,h in inputs.items()):raise RuntimeError('Frozen input changed')
        worker=dict(spec)
        if name=='hold':worker['hold_start_seconds']=start
        f=out/f'{name}_plan.json';f.write_text(json.dumps(worker,indent=2))
        print(f'{name}: new process; log={out/name}.log',flush=True)
        with (out/f'{name}.log').open('w') as log:
            subprocess.run(['bash',str(ROOT/'scripts/isaacsim_preload_hold_case.sh'),'--plan',str(f),'--output',str(out/name)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        results[name]=json.loads((out/name/'result.json').read_text())
        with np.load(out/name/'trajectory.npz') as z:trajectories[name]={k:z[k] for k in z.files}
        with np.load(out/name/'contact_points.npz') as z:
            timing[name]=summarize_contact_timing(trajectories[name],[str(x) for x in z['link_names']],[str(x) for x in z['finger_names']],results[name]['info']['cases'].index('gy+'))
    with np.load(source/'policy/trajectory.npz') as old:
        repeat=compare_traces(old,trajectories['baseline'],prefix_seconds=gravity+cfg['gravity_seconds'])
    prefix=compare_traces(trajectories['baseline'],trajectories['hold'],prefix_seconds=start)
    t=trajectories['hold'];ids=np.asarray(t['independent_ids'],int);anchor=np.argmin(abs(t['time']-start))
    mask=(t['time']>=start)&(t['time']<=gravity)
    offsets=t['joint_target'][:,:,ids]-t['joint_reference'][:,:,ids]
    max_change=float(np.max(abs(offsets[mask]-offsets[anchor])))
    if any(digest(f)!=h for f,h in inputs.items()):raise RuntimeError('Frozen input changed during evaluation')
    verdict=dict(baseline_reproduced=repeat,intervention_prefix=prefix,hold_max_offset_change_rad=max_change,
        held_offsets_constant=max_change<=1e-6,results={n:r['info'] for n,r in results.items()},first_failure={n:r['first_failure'] for n,r in results.items()},contact_timing=timing)
    gate=repeat['passed'] and prefix['passed'] and verdict['held_offsets_constant']
    verdict['causal_comparison_allowed']=bool(gate)
    (out/'comparison.json').write_text(json.dumps(verdict,indent=2))
    lines=['# 预加载已施加偏置保持消融','',f'保持区间：{start}–{gravity}s；之后连续恢复原策略。原参考、检查点、限速和验收不变。',
        f'原策略完整复现：{repeat["passed"]}；干预前前缀一致：{prefix["passed"]}；偏置保持最大变化：{max_change:.6g}rad；可比较：{gate}。','',
        '|方向|原策略|保持变体|原首次失败|变体首次失败|','|---|---|---|---|---|']
    for i,c in enumerate(results['baseline']['info']['cases']):
        lines.append(f'|{c}|{results["baseline"]["info"]["passed"][i]}|{results["hold"]["info"]["passed"][i]}|{results["baseline"]["first_failure"][i]}|{results["hold"]["first_failure"][i]}|')
    for n,s in timing.items():lines += ['',f'{n} gy+：{s["classification"]}；预加载峰值推移 {s["preload_peak_displacement_mm"]:.4f}mm；3秒各指力 {s["snapshots"][3]["fingers"]}。']
    lines += ['', '这是评估控制消融，不是重新训练。过程门控失败时不可归因；单组改善后仍须检验其他组成功保留。连续误差及mimic原始指标见comparison.json。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'status.json').write_text(json.dumps(dict(status='complete',causal_comparison_allowed=gate)))
    print(out/'REPORT.md')
    if not gate:raise SystemExit('Process/hold checks failed; inspect comparison.json')


if __name__=='__main__':main()
