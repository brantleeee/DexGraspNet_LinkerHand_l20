"""Checkpoint-free reference baseline in the unchanged current v4 RL environment."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.paired_evaluation import make_samples,validate_samples,FixedUniform
from isaacsim_grasp.reference_workflow import FirstFailureRecorder,admission
from evaluate_isaacsim_residual_paired import initial_state,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',default='jenga',choices=['mug','jenga','banana','bottle','camera'])
    p.add_argument('--candidate',type=int,default=7)
    p.add_argument('--task-version',choices=['v4','v5'],default='v4')
    p.add_argument('--preload-seconds',type=float,default=.5)
    p.add_argument('--samples',type=Path,help='Reuse an exact saved translation deck')
    p.add_argument('--reference-results',type=Path,default=ROOT/'data/experiments/l20_five_objects_acquire_10s_20261105')
    p.add_argument('--num-envs',type=int,default=6);p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--position-noise',type=float,default=.0005);p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() and not a.plan_only:p.error('Output must be new')
    samples=json.loads(a.samples.read_text()) if a.samples else make_samples(a.episodes,a.num_envs,a.position_noise,a.seed)
    offsets=validate_samples(samples,a.episodes,a.num_envs,a.position_noise)
    command=[str(ROOT/'.venv_l20/bin/python'),str(ROOT/f'scripts/run_isaacsim_residual_{a.task_version}.py'),
        '--mode','plan','--object',a.object,'--candidate',str(a.candidate),'--reference-results',str(a.reference_results.resolve()),
        '--num-envs',str(a.num_envs),'--position-noise',str(a.position_noise),'--output',str(out)]
    if a.task_version=='v5':command+=['--preload-seconds',str(a.preload_seconds)]
    plan=json.loads(subprocess.run(command,check=True,capture_output=True,text=True,cwd=ROOT).stdout)
    config=plan['config'];ref=plan['reference'];plan.update(baseline_only=True,samples=samples)
    if a.plan_only:print(json.dumps(plan,indent=2));return
    out.mkdir(parents=True);save(out/'plan.json',plan);save(out/'config.json',config);save(out/'perturbations.json',samples)
    shutil.copy2(ref['scene'],out/'reference_scene.usda');ref=dict(ref,scene=str(out/'reference_scene.usda'));save(out/'reference_snapshot.json',ref)
    save(out/'status.json',dict(status='starting'))
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,multi_gpu=False,sync_loads=False,create_new_stage=False,limit_cpu_threads=8,anti_aliasing=2));env=None
    try:
        if a.task_version=='v5':
            from isaacsim_grasp.environment_v5 import ResidualGraspEnv
        else:
            from isaacsim_grasp.environment_v4 import ResidualGraspEnv
        from isaacsim_grasp.diagnostics import PhysicsDiagnostics
        env=ResidualGraspEnv(ref,config,app);tracker=FirstFailureRecorder(env.history);recorder=PhysicsDiagnostics(env)
        rootpos,rootquat=env.hand.get_world_poses();rootpos=rootpos.copy();rootquat=rootquat.copy();rows=[];states={}
        result=dict(complete=False,reference=dict(object=a.object,candidate=a.candidate,scale=ref['scale']),controller='zero_residual',trials=rows)
        for episode in range(a.episodes):
            env.hand.set_world_poses(rootpos,rootquat);env.hand.set_velocities(np.zeros((env.n,6),np.float32))
            env.rng=FixedUniform(offsets[episode],a.position_noise);env.reset();tracker.begin();recorder.begin()
            for key,value in initial_state(env).items():states[f'zero_residual_episode_{episode:03d}_{key}']=value
            while True:
                obs,reward,done,info=env.step(np.zeros((env.n,16),np.float32))
                if done:break
            diagfile=Path('diagnostics')/f'zero_residual_batch_{episode:03d}.npz';recorder.finish(out/diagfile)
            for i,case in enumerate(info['cases']):
                rows.append(dict(episode=episode,env=i,group=episode*(env.n//6)+i//6,case=case,offset_m=offsets[episode,i].tolist(),
                    passed=info['passed'][i],acquired=info['acquired'][i],first_failure=tracker.first[i],
                    max_slip_m=info['max_slip_m'][i],max_angle_deg=info['max_angle_deg'][i],max_penetration_m=info['max_penetration_m'][i],
                    max_mimic_error_rad=info['max_mimic_error_rad'][i]))
            save(out/'baseline.json',result);save(out/'status.json',dict(status='evaluating_baseline',completed_batches=episode+1))
            print(f'zero_residual baseline: {episode+1}/{a.episodes}, passed={sum(info["passed"])}/{env.n}',flush=True)
        result['admission']=admission(rows);result['complete']=True
        save(out/'baseline.json',result);save(out/'admission.json',result['admission']);save(out/'physics_audit.json',env.audit)
        np.savez_compressed(out/'initial_states.npz',**states)
        with (out/'trials.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        lines=['# 当前RL环境零残差基线','',f"参考：{a.object} 原始下标{a.candidate}，scale={ref['scale']}；无检查点，无策略动作。",'',
            '|组|方向|通过|首次失败时间s|阶段|原因|','|---|---|---|---:|---|---|']
        for r in rows:
            fail=r['first_failure'];lines.append(f"|{r['group']}|{r['case']}|{r['passed']}|{fail['time_seconds'] if fail else '—'}|{fail['phase'] if fail else '—'}|{','.join(fail['reasons']) if fail else '—'}|")
        lines+=['',f"训练准入：{result['admission']}。",'只有准备阶段全部合格且至少一个重力阶段失败，才适合本次重力残差修复实验；准入不保证训练成功。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n');save(out/'status.json',dict(status='complete'))
        print(json.dumps(result['admission'],ensure_ascii=False,indent=2));print(f'Complete: {out}',flush=True)
    except BaseException as exc:
        import traceback
        save(out/'error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));save(out/'status.json',dict(status='failed'));raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
