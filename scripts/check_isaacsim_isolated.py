"""Fresh-process zero-zero reproducibility, per-finger diagnosis and optional policy."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.isolated import select_group,compare_traces
from isaacsim_grasp.reference import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--group',type=int,default=18)
    p.add_argument('--object',default='mug');p.add_argument('--candidate',type=int,default=0)
    p.add_argument('--reference-results',type=Path)
    p.add_argument('--checkpoint',type=Path,help='Optional third fresh process for policy, after zero-zero passes')
    p.add_argument('--variant-reference',type=Path,help='Validate a saved derived reference on another translation; no old policy reuse')
    p.add_argument('--preload-gain',type=float,default=.5)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    from isaacsim_grasp.evaluation_interventions import validate_preload_gain
    try:validate_preload_gain(a.preload_gain)
    except ValueError as exc:p.error(str(exc))
    if out.exists() and not a.plan_only:p.error('Output must be new')
    samples=select_group(json.loads(a.samples.read_text()),a.group)
    command=[sys.executable,str(ROOT/'scripts/run_isaacsim_residual_v5.py'),'--mode','plan','--object',a.object,
        '--candidate',str(a.candidate),'--num-envs','6','--position-noise',str(samples['amplitude_m']),'--output',str(out)]
    if a.checkpoint:command+=['--checkpoint',str(a.checkpoint.resolve())]
    if a.reference_results:command+=['--reference-results',str(a.reference_results.resolve())]
    plan=json.loads(subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True).stdout)
    spec=dict(reference=plan['reference'],config=plan['config'],samples=samples,controller='zero_residual',
              source_samples=str(a.samples.resolve()),source_samples_sha256=digest(a.samples))
    if a.variant_reference:
        if a.checkpoint:p.error('A changed reference requires a new policy; do not reuse this checkpoint')
        from isaacsim_grasp.reference_variants import variant_targets
        saved=json.loads(a.variant_reference.read_text());variant=saved['derived_variant']
        if variant_targets(plan['reference'],variant)!=saved['targets']:p.error('Derived reference target mismatch')
        spec.update(variant=variant,variant_source=str(a.variant_reference.resolve()),variant_source_sha256=digest(a.variant_reference))
    if a.plan_only:print(json.dumps(spec,indent=2));return
    out.mkdir(parents=True);(out/'worker_plan.json').write_text(json.dumps(spec,indent=2));(out/'perturbations.json').write_text(json.dumps(samples,indent=2))
    def run(name,worker):
        path=out/f'{name}_plan.json';path.write_text(json.dumps(worker,indent=2))
        with (out/f'{name}.log').open('w') as log:
            subprocess.run(['bash',str(ROOT/'scripts/isaacsim_isolated_case.sh'),'--plan',str(path),'--output',str(out/name)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status=json.loads((out/name/'status.json').read_text())
        if status['status']!='complete':raise RuntimeError(f'Incomplete: {out/name}')
    run('zero_a',spec);run('zero_b',spec)
    with np.load(out/'zero_a/trajectory.npz') as x,np.load(out/'zero_b/trajectory.npz') as y:
        prefix=compare_traces(x,y);full=compare_traces(x,y,prefix_seconds=13.)
    ra=json.loads((out/'zero_a/result.json').read_text());rb=json.loads((out/'zero_b/result.json').read_text())
    verdict=dict(complete=True,zero_zero_prefix=prefix,zero_zero_full=full,
        zero_zero_pass_flags_identical=ra['info']['passed']==rb['info']['passed'],
        different_os_processes=ra['pid']!=rb['pid'],source_group=a.group,policy_evaluated=False)
    # Full zero-zero trajectories must be repeatable before attributing outcomes.
    gate=prefix['passed'] and full['passed'] and verdict['zero_zero_pass_flags_identical'] and verdict['different_os_processes']
    verdict['process_reproducible']=bool(gate)
    verdict['reference_refinement_admitted']=bool(gate and not all(ra['info']['passed']))
    if a.checkpoint and gate:
        spec=dict(spec,controller='policy',checkpoint=str(a.checkpoint.resolve()),preload_gain=a.preload_gain)
        run('policy',spec)
        with np.load(out/'zero_a/trajectory.npz') as x,np.load(out/'policy/trajectory.npz') as y:
            verdict['zero_policy_prefix']=compare_traces(x,y)
        verdict['policy_evaluated']=True
    (out/'reproducibility.json').write_text(json.dumps(verdict,indent=2))
    lines=['# 独立进程零残差复现检查','',f'源扰动组：{a.group}；每个进程独立构建场景、仅运行一回合六方向。','',
        f"不同OS进程：{verdict['different_os_processes']}；2.5秒前缀复现：{prefix['passed']}；完整13秒复现：{full['passed']}；标签一致：{verdict['zero_zero_pass_flags_identical']}。",'',
        f"局部参考细化准入：{verdict['reference_refinement_admitted']}（需要过程复现且存在真实失败）。数值过程容差不改变物理抓取验收。",'',
        '|方向|zero_a|zero_b|首次失败zero_a|','|---|---|---|---|']
    for i,case in enumerate(ra['info']['cases']):lines.append(f"|{case}|{ra['info']['passed'][i]}|{rb['info']['passed'][i]}|{ra['first_failure'][i]}|")
    lines+=['','各指接触事件见 zero_a/contact_events.json；点、法向、法向力、摩擦锚点见 contact_points.npz；轨迹含逐指/逐连杆力。',
        '若复现未通过，先修复实验过程，不启动自动细化或策略训练。']
    if verdict.get('zero_policy_prefix'):lines+=['',f"零残差/策略前缀：{verdict['zero_policy_prefix']}。"]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/plot_isolated_contacts.py'),'--run',str(out/'zero_a')],cwd=ROOT,check=True)
    (out/'status.json').write_text(json.dumps(dict(status='complete',process_reproducible=gate,admitted=verdict['reference_refinement_admitted'])))
    print(json.dumps(verdict,indent=2));print(f'Complete: {out}')


if __name__=='__main__':main()
