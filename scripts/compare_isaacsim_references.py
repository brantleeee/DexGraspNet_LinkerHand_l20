"""Compare original/derived references on new samples; four fresh processes/group."""
import argparse,html,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.paired_evaluation import make_samples
from isaacsim_grasp.reference import load_reference,digest
from isaacsim_grasp.reference_variants import variant_targets
from isaacsim_grasp.reference_comparison import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant-reference',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=5);p.add_argument('--position-noise',type=float,default=.0005)
    p.add_argument('--seed',type=int,default=20261113);p.add_argument('--plan-only',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if out.exists() and not a.plan_only:p.error('Output must be new')
    if a.episodes<1:p.error('Positive episodes required')
    saved=json.loads(a.variant_reference.read_text());ref=load_reference(saved['source_results'],saved['object'],saved['candidate'])
    if variant_targets(ref,saved['derived_variant'])!=saved['targets']:p.error('Derived target/parent mismatch')
    samples=make_samples(a.episodes,6,a.position_noise,a.seed)
    plan=dict(variant_reference=str(a.variant_reference.resolve()),variant_sha256=digest(a.variant_reference),
        parent_scene_sha256=ref['scene_sha256'],reference_results=ref['source_results'],samples=samples,
        object=ref['object'],candidate=ref['candidate'],native_processes=4*a.episodes,
        comparisons='original reference zero residual vs refined reference zero residual; two fresh-process repeats each')
    if a.plan_only:print(json.dumps(plan,indent=2));return
    out.mkdir(parents=True);(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'perturbations.json').write_text(json.dumps(samples,indent=2))
    rows=[]
    try:
        for group in range(a.episodes):
            results={}
            for mode in ['original','refined']:
                if digest(a.variant_reference)!=plan['variant_sha256']:raise ValueError('Candidate changed during validation')
                folder=out/f'group_{group:03d}'/mode;folder.parent.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,str(ROOT/'scripts/check_isaacsim_isolated.py'),'--samples',str(out/'perturbations.json'),
                    '--group',str(group),'--object',ref['object'],'--candidate',str(ref['candidate']),
                    '--reference-results',ref['source_results'],'--output',str(folder)]
                if mode=='refined':command+=['--variant-reference',str(a.variant_reference.resolve())]
                logpath=folder.parent/f'{mode}.log'
                print(f'[reference suite] sample {group+1}/{a.episodes} {mode}; log={logpath}',flush=True)
                (out/'status.json').write_text(json.dumps(dict(status='running',group=group,reference=mode,log=str(logpath))))
                with logpath.open('w') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                verdict=json.loads((folder/'reproducibility.json').read_text())
                if not verdict['process_reproducible']:raise RuntimeError(f'Process repeatability failed: {folder}')
                results[mode]=json.loads((folder/'zero_a/result.json').read_text())
            original=results['original'];refined=results['refined']
            if original['info']['cases']!=refined['info']['cases']:raise ValueError('Direction ordering differs')
            base=out/f'group_{group:03d}'
            with np.load(base/'original/zero_a/initial_states.npz') as x,np.load(base/'refined/zero_a/initial_states.npz') as y:
                for key in ['object_local_position','object_local_quaternion','object_local_linear_velocity','object_local_angular_velocity']:
                    np.testing.assert_allclose(x[key],y[key],rtol=0,atol=1e-7,err_msg=f'Initial object states differ: {key}')
            for i,case in enumerate(original['info']['cases']):
                rows.append(dict(group=group,case=case,offset_m=samples['offsets_m'][group][i],
                    original_passed=original['info']['passed'][i],refined_passed=refined['info']['passed'][i],
                    original_failure=original['first_failure'][i],refined_failure=refined['first_failure'][i],
                    original_metrics={k:original['info'][k][i] for k in ['max_slip_m','max_angle_deg','max_penetration_m','max_mimic_error_rad']},
                    refined_metrics={k:refined['info'][k][i] for k in ['max_slip_m','max_angle_deg','max_penetration_m','max_mimic_error_rad']}))
            (out/'comparison.json').write_text(json.dumps(dict(complete=False,rows=rows),indent=2))
            print(f'[reference suite] sample {group+1}: original={sum(original["info"]["passed"])}/6 refined={sum(refined["info"]["passed"])}/6',flush=True)
        summary=summarize(rows);(out/'comparison.json').write_text(json.dumps(dict(complete=True,summary=summary,rows=rows),indent=2))
        lines=['# 原参考与MCP改良参考：新样本独立进程对照','',f"固定候选：{a.variant_reference.resolve()}。seed={a.seed}，每轴±{a.position_noise*1000:g}mm。",'',
            f"每种参考{summary['trials_per_reference']}次方向试验，独立平移{summary['translation_samples']}个。每个参考/平移两次全新进程复现，重复不额外计入样本量。",'',
            f"方向通过：原参考{summary['original_passed']}/{summary['trials_per_reference']}；改良{summary['refined_passed']}/{summary['trials_per_reference']}。",
            f"六方向全通过：原参考{summary['original_all_six']}/{a.episodes}；改良{summary['refined_all_six']}/{a.episodes}。",
            f"配对：{summary['outcomes']}。refined_only为修复，original_only为损害。",'',
            '这是静态参考细化的效果，不是RL策略增益。先看修复、损害与完整六方向，再检查连续误差。',
            '每个参考自身须复现；两个参考的初始关节目标有意不同，不能要求它们的闭合轨迹相同。物体初态逐项一致。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        links=[]
        for g in range(a.episodes):
            for mode in ['original','refined']:
                rel=f'group_{g:03d}/{mode}/zero_a/contacts/index.html';links.append(f'<li><a href="{rel}">样本{g} {mode}</a></li>')
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>原参考/改良参考独立进程对照</h1><pre>'+html.escape('\n'.join(lines))+'</pre><ul>'+''.join(links)+'</ul>')
        (out/'status.json').write_text(json.dumps(dict(status='complete')));print(f'Complete: {out}',flush=True)
    except BaseException as exc:
        (out/'status.json').write_text(json.dumps(dict(status='failed',message=str(exc))));raise


if __name__=='__main__':main()
