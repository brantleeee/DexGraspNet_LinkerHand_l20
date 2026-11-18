"""Evaluate v2/v3/v4 checkpoints on explicit paired samples with read-only action diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from isaacsim_grasp.reference import load_reference,digest
from isaacsim_grasp.paired_evaluation import CASES,make_samples,validate_samples,FixedUniform,compare_initial_states,summarize_pairs
from run_isaacsim_residual_v2 import implementation_hashes


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def report(out,summary):
    table=summary.get('paired')
    if not table:return
    lines=[f"# {summary.get('policy_version','v2')} 初始位置扰动配对评估",'',
        f"每轴均匀扰动 ±{summary['position_noise_m']*1000:g} mm；每组控制 {table['trials']} 次方向试验、{table['six_direction_groups']} 个完整六方向扰动样本。",'',
        '两组读取同一份 perturbations.json；显式初始状态逐项检查一致，其他物理与控制配置沿用检查点。',
        '数值、成功条件、质量、摩擦、增益和时长均未改变。', '',
        '|方向|试验数/控制组|零残差通过|策略通过|','|---|---:|---:|---:|']
    for case,d in table['per_direction'].items():lines.append(f"|{case}|{d['trials']}|{d['zero_passed']}/{d['trials']}|{d['policy_passed']}/{d['trials']}|")
    lines+=['',f"逐方向配对结果：{table['outcomes']}。",
        f"完整六方向全通过：零残差 {sum(g['zero_all_six'] for g in table['groups'])}/{table['six_direction_groups']}；策略 {sum(g['policy_all_six'] for g in table['groups'])}/{table['six_direction_groups']}。",'',
        'policy_only 是该扰动/方向下策略通过而基线未通过；zero_only 则是策略破坏了基线能力。',
        f"本次有{table['six_direction_groups']}个平移样本；同一平移的六方向相关，不能把方向试验数量当成独立平移样本数。小样本只作初步筛查。",
        '此配对固定显式初态，不声称清除PhysX全部内部warm-start状态或保证逐位一致的后续轨迹。',
        '原始样本、实际初态、逐试验滑移/转角/穿透/mimic误差见JSON和CSV。']
    if summary.get('baseline_recheck'):
        lines+=['',f"独立基线复核：{summary['baseline_recheck']}。",
                '先确认零残差标签复现，再将policy_only解释为本组原失败方向的修复；标签未复现时先排查环境。']
    if summary.get('preload_seconds') is not None:
        lines+=['',f"v5残差在零重力稳定阶段最后{summary['preload_seconds']}秒及重力阶段生效，闭合参考、奖励时序与严格验收沿用v4。",
                '每次方向试验的preload_response记录开重力前的力、穿透、关节运动和mimic误差；不能仅凭接触力增大认定抓取稳定。']
    if summary.get('evaluation_intervention'):
        lines+=['',f"评估专用干预：{summary['evaluation_intervention']}。这是同一检查点的控制消融，不是重新训练或原策略原样输出。",
                'policy_raw_action/policy_requested_residual保存原策略输出；raw_action/requested_residual是送入环境的命令。限速后的实际偏置仍见applied_residual。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if summary.get('diagnostics'):
        with (out/'REPORT.md').open('a') as f:f.write('\n[统一位置、速度、力矩大图](diagnostics.html)。原始240Hz数据见 diagnostics/*.npz。\n')


def initial_state(env):
    pos,quat,q,qd,linear,angular=env.state();rootpos,rootquat=env.hand.get_world_poses()
    return dict(object_local_position=pos.copy(),object_local_quaternion=quat.copy(),joint_position=q.copy(),joint_velocity=qd.copy(),
        object_local_linear_velocity=linear.copy(),object_local_angular_velocity=angular.copy(),
        hand_root_position=rootpos.copy(),hand_root_quaternion=rootquat.copy(),hand_root_velocity=env.hand.get_velocities().copy(),
        filtered_residual=env.controller.offset.copy(),joint_target=env.last_target.copy(),observation=env.observe().copy())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=6)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--position-noise',type=float,default=.0005)
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--samples',type=Path,help='Reuse an existing perturbations.json; dimensions and bounds must match')
    p.add_argument('--compare-results',type=Path,help='After evaluation compare continuous margins against this saved same-sample run')
    p.add_argument('--baseline-results',type=Path,help='Require this reference baseline and its identical saved perturbations/explicit initial states')
    p.add_argument('--preload-gain',type=float,help='v5 evaluation only: multiply tanh-space requests in the zero-gravity preload window by [0,1]')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--check-resets-only',action='store_true',help='Verify paired explicit resets without advancing physics; not a stability evaluation')
    p.add_argument('--skip-plot',action='store_true',help='Record full diagnostics but render the HTML later with the project Python')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();checkpoint=a.checkpoint.resolve()
    if out.exists() and not a.plan_only:p.error('Output must be new')
    if a.num_envs<6 or a.num_envs%6 or a.episodes<1 or not np.isfinite(a.position_noise) or a.position_noise<0:p.error('Invalid dimensions or position-noise')
    if a.plan_only and a.check_resets_only:p.error('Choose plan-only or check-resets-only')
    training_config=json.loads((checkpoint.parent/'config.json').read_text())
    intervention=None
    if a.preload_gain is not None:
        from isaacsim_grasp.evaluation_interventions import validate_preload_gain
        try:validate_preload_gain(a.preload_gain)
        except ValueError as exc:p.error(str(exc))
        if training_config['version']!='isaacsim_sdf_residual_v5':p.error('--preload-gain requires a v5 checkpoint')
        intervention=dict(type='preload_request_multiplier',preload_gain=a.preload_gain,
            policy_only=True,tanh_space=True,gravity_gain=1.,rate_limiter_unchanged=True,
            offset_continuous_at_gravity=True,not_retrained=True)
    if training_config['version']=='isaacsim_sdf_residual_v5':
        from run_isaacsim_residual_v5 import implementation_hashes as check_hashes
    elif training_config['version']=='isaacsim_sdf_residual_v4':
        from run_isaacsim_residual_v4 import implementation_hashes as check_hashes
    elif training_config['version']=='isaacsim_sdf_residual_v3':
        from run_isaacsim_residual_v3 import implementation_hashes as check_hashes
    elif training_config['version']=='isaacsim_sdf_residual_v2':check_hashes=implementation_hashes
    else:p.error('This evaluator requires v2, v3, v4 or v5')
    if training_config.get('implementation_sha256')!=check_hashes():p.error('Training implementation changed; refusing to bypass checkpoint compatibility')
    old=json.loads((checkpoint.parent/'reference_snapshot.json').read_text())
    ref=load_reference(old['source_results'],old['object'],old['candidate'])
    if ref['scene_sha256']!=old['scene_sha256'] or ref['targets']!=old['targets']:p.error('Reference changed')
    sample_file=a.samples or (a.baseline_results/'perturbations.json' if a.baseline_results else None)
    samples=json.loads(sample_file.read_text()) if sample_file else make_samples(a.episodes,a.num_envs,a.position_noise,a.seed)
    offsets=validate_samples(samples,a.episodes,a.num_envs,a.position_noise)
    config=dict(training_config,num_envs=a.num_envs,position_noise=a.position_noise)
    changes={k:dict(training=training_config.get(k),evaluation=v) for k,v in config.items() if training_config.get(k)!=v}
    if set(changes)-{'num_envs','position_noise'}:raise ValueError('Unexpected physics/control changes')
    baseline=None
    if a.baseline_results:
        from isaacsim_grasp.reference_workflow import validate_baseline
        baseline=validate_baseline(a.baseline_results,ref,config,samples)
    if a.compare_results:
        previous=json.loads((a.compare_results/'evaluation.json').read_text())
        previous_samples=json.loads((a.compare_results/'perturbations.json').read_text())
        previous_config=json.loads((a.compare_results/'config.json').read_text())
        previous_ref=json.loads((a.compare_results/'reference_snapshot.json').read_text())
        if not previous['complete'] or previous_ref['scene_sha256']!=ref['scene_sha256']:p.error('Comparison run incomplete or uses a different scene')
        if samples['offsets_m']!=previous_samples['offsets_m']:p.error('Comparison requires exactly the previous saved perturbations')
        for key in ['residual_scale','residual_rate','kp','kd','physics_dt','decimation','close_seconds','settle_seconds','gravity_seconds','open_fraction','collision','mimic','position_noise','num_envs']:
            if config[key]!=previous_config[key]:p.error(f'Comparison setting differs: {key}')
    plan=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),training_config=training_config,evaluation_config=config,
        permitted_configuration_changes=changes,reference=ref,episodes=a.episodes,trials_per_mode=a.episodes*a.num_envs,
        six_direction_groups=a.episodes*a.num_envs//6,samples=samples,
        note='Only initial object translation is randomized; no rotation/mass/friction/gain/time randomization.')
    if intervention is not None:plan['evaluation_intervention']=intervention
    if a.plan_only:print(json.dumps(plan,indent=2));return
    out.mkdir(parents=True);save(out/'plan.json',plan);save(out/'perturbations.json',samples)
    save(out/'status.json',dict(status='starting'))
    shutil.copy2(ref['scene'],out/'reference_scene.usda');ref=dict(ref,scene=str(out/'reference_scene.usda'))
    save(out/'reference_snapshot.json',ref);save(out/'config.json',config)
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,multi_gpu=False,sync_loads=False,create_new_stage=False,limit_cpu_threads=8,anti_aliasing=2))
    env=None
    try:
        import torch
        if config['version']=='isaacsim_sdf_residual_v5':
            from isaacsim_grasp.environment_v5 import ResidualGraspEnv
        elif config['version']=='isaacsim_sdf_residual_v4':
            from isaacsim_grasp.environment_v4 import ResidualGraspEnv
        elif config['version']=='isaacsim_sdf_residual_v3':
            from isaacsim_grasp.environment_v3 import ResidualGraspEnv
        else:
            from isaacsim_grasp.environment_v2 import ResidualGraspEnv
        from isaacsim_grasp.ppo import ActorCritic
        from isaacsim_grasp.diagnostics import PhysicsDiagnostics,TORQUE_NOTE
        from isaacsim_grasp.reference_workflow import FirstFailureRecorder
        env=ResidualGraspEnv(ref,config,app)
        failure_tracker=FirstFailureRecorder(env.history)
        if intervention is not None:
            from isaacsim_grasp.evaluation_interventions import InterventionDiagnostics,scale_preload_action
        recorder=None if a.check_resets_only else (InterventionDiagnostics(env) if intervention is not None else PhysicsDiagnostics(env))
        rootpos,rootquat=env.hand.get_world_poses();rootpos=rootpos.copy();rootquat=rootquat.copy()
        saved_states={};actual_states={};differences={};readback_errors={};summary=dict(complete=False,checkpoint=str(checkpoint),reference=dict(object=ref['object'],candidate=ref['candidate'],scale=ref['scale']),
            policy_version=config['version'],
            position_noise_m=a.position_noise,sample_seed=samples['seed'],episodes=a.episodes,num_envs=a.num_envs,
            perturbations_sha256=digest(out/'perturbations.json'),runs={},explicit_initial_states_identical=False,
            configuration_changes=changes,check_resets_only=a.check_resets_only)
        if config['version']=='isaacsim_sdf_residual_v5':summary['preload_seconds']=config['preload_seconds']
        if intervention is not None:summary['evaluation_intervention']=intervention
        if recorder is not None:
            summary['diagnostics']={'zero_residual':[],'policy':[]}
            summary['diagnostics_metadata']=dict(sample_hz=240,all_physics_steps=True,torque_note=TORQUE_NOTE,
                action_layers='raw_action -> requested_residual -> effective_requested_residual -> applied_residual; actual joint position/velocity recorded separately',
                initial_measured_effort='unavailable_before_first_physics_step_recorded_as_NaN')
        model=None
        if not a.check_resets_only:
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            ckpt=torch.load(checkpoint,map_location='cuda:0',weights_only=False)
            if ckpt['version']!=config['version'] or ckpt['scene_sha256']!=ref['scene_sha256']:raise ValueError('Checkpoint content identity mismatch')
            model=ActorCritic(ckpt['obs_dim'],16).to('cuda:0');model.load_state_dict(ckpt['model']);model.eval()
        for mode in ['zero_residual','policy']:
            rows=[]
            for episode in range(a.episodes):
                env.hand.set_world_poses(rootpos,rootquat);env.hand.set_velocities(np.zeros((env.n,6),np.float32))
                env.rng=FixedUniform(offsets[episode],a.position_noise)
                obs=env.reset();failure_tracker.begin();state=initial_state(env)
                if baseline is not None and mode=='zero_residual':
                    with np.load(a.baseline_results/'initial_states.npz') as saved:
                        prefix=f'zero_residual_episode_{episode:03d}_'
                        expected={name[len(prefix):]:saved[name] for name in saved.files if name.startswith(prefix)}
                    compare_initial_states(expected,state)
                readback_error=float(np.max(np.abs(state['object_local_position']-offsets[episode])))
                if readback_error>2e-6:raise RuntimeError('Actual object reset does not match requested perturbation within 2 micrometers')
                readback_errors[f'{mode}_{episode}']=readback_error
                if mode=='zero_residual':saved_states[episode]=state
                else:differences[str(episode)]=compare_initial_states(saved_states[episode],state)
                for key,value in state.items():actual_states[f'{mode}_episode_{episode:03d}_{key}']=value
                if a.check_resets_only:continue
                recorder.begin()
                while True:
                    with torch.inference_mode():
                        action=np.zeros((env.n,16),np.float32) if mode=='zero_residual' else model.actor(torch.as_tensor(obs,device='cuda:0')).cpu().numpy()
                    if intervention is not None:
                        original_action=action.copy()
                        action,multiplier=scale_preload_action(action,a.preload_gain if mode=='policy' else 1.,
                            env.physics_step,env.action_start,env.gravity_start)
                        recorder.set_policy_action(original_action,multiplier)
                    obs,reward,done,info=env.step(action)
                    if done:break
                diagfile=Path('diagnostics')/f'{mode}_batch_{episode:03d}.npz'
                diagnostics=recorder.finish(out/diagfile);summary['diagnostics'][mode].append(str(diagfile))
                if config['version']=='isaacsim_sdf_residual_v5':
                    from isaacsim_grasp.preload import response_summary
                    with np.load(out/diagfile) as recorded:
                        preload=response_summary(recorded,config['preload_seconds'],env.controller.mimics)
                else:preload=None
                for i,case in enumerate(info['cases']):
                    rows.append(dict(episode=episode,env=i,group=episode*(env.n//6)+i//6,case=case,
                        offset_m=offsets[episode,i].tolist(),passed=bool(info['passed'][i]),
                        max_slip_m=info['max_slip_m'][i],max_angle_deg=info['max_angle_deg'][i],max_penetration_m=info['max_penetration_m'][i],
                        max_mimic_error_rad=info['max_mimic_error_rad'][i],acquired=info['acquired'][i],
                        last_second_contact=info['last_second_contact'][i],failure_reasons=[k for k,v in info['failure_reasons'].items() if v[i]]))
                    rows[-1]['action_summary']=diagnostics['action_summary'][i]
                    rows[-1]['first_failure']=failure_tracker.first[i]
                    if preload is not None:rows[-1]['preload_response']=preload[i]
                save(out/'status.json',dict(status='evaluating',mode=mode,completed_batches=episode+1,total_batches=a.episodes))
                print(f'{mode}: batch {episode+1}/{a.episodes}, passed={sum(info["passed"])}/{env.n}',flush=True)
            summary['runs'][mode]=rows
            save(out/'evaluation.json',summary)
        summary['explicit_initial_states_identical']=True;summary['initial_state_max_abs_differences']=differences
        summary['requested_position_readback_max_error_m']=readback_errors
        if not a.check_resets_only:summary['paired']=summarize_pairs(summary['runs']['zero_residual'],summary['runs']['policy'])
        summary['complete']=True
        if baseline is not None:
            old={(r['episode'],r['env']):r for r in baseline['trials']}
            summary['baseline_recheck']=dict(source=str(a.baseline_results.resolve()),explicit_initial_states_identical=True,
                pass_flags_reproduced=all(r['passed']==old[(r['episode'],r['env'])]['passed'] for r in summary['runs']['zero_residual']))
        np.savez_compressed(out/'initial_states.npz',**actual_states)
        save(out/'evaluation.json',summary);save(out/'physics_audit.json',env.audit)
        save(out/'evaluator_sources.json',{str(path.relative_to(ROOT)):digest(path) for path in [Path(__file__),ROOT/'isaacsim_grasp/paired_evaluation.py',ROOT/'isaacsim_grasp/diagnostics.py',ROOT/'isaacsim_grasp/action_analysis.py',ROOT/'scripts/plot_isaacsim_paired_diagnostics.py',ROOT/'scripts/compare_isaacsim_margin.py']})
        if intervention is not None:
            sources=json.loads((out/'evaluator_sources.json').read_text())
            sources['isaacsim_grasp/evaluation_interventions.py']=digest(ROOT/'isaacsim_grasp/evaluation_interventions.py')
            save(out/'evaluator_sources.json',sources)
        if not a.check_resets_only:
            combined=[dict(control=name,**row,offset_x_m=row['offset_m'][0],offset_y_m=row['offset_m'][1],offset_z_m=row['offset_m'][2]) for name,rows in summary['runs'].items() for row in rows]
            fields=[k for k in combined[0] if k!='offset_m']
            with (out/'trials.csv').open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(combined)
            report(out,summary)
            if a.compare_results:
                from compare_isaacsim_margin import compare
                compare(out,a.compare_results)
            if not a.skip_plot:
                subprocess.run([str(ROOT/'.venv_l20/bin/python'),str(ROOT/'scripts/plot_isaacsim_paired_diagnostics.py'),
                                '--results',str(out)],cwd=ROOT,check=True)
        save(out/'status.json',dict(status='complete',check_resets_only=a.check_resets_only))
        print(f'Complete: {out}',flush=True)
    except BaseException as exc:
        import traceback
        save(out/'error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        save(out/'status.json',dict(status='failed'));raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
