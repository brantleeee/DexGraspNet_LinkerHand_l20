"""v2: gravity-only residual PPO with explicit strict history and finite-horizon terminals."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaacsim_grasp.reference import load_reference,digest

DEFAULTS=dict(mug=20,jenga=7,banana=13,bottle=2,camera=10)


def implementation_hashes():
    paths=[Path(__file__),*[ROOT/'isaacsim_grasp'/f'{name}.py' for name in ['environment','environment_v2','history','control','ppo']]]
    return {str(path.relative_to(ROOT)):digest(path) for path in paths}


class Metrics:
    def __init__(self,output):
        self.file=(output/'metrics.csv').open('w',newline='');self.csv=csv.writer(self.file)
        self.csv.writerow(['iteration','tag','value']);self.tb=None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb=SummaryWriter(str(output))
        except ImportError as exc:
            print(f'TensorBoard unavailable ({exc}); scalar metrics are saved to metrics.csv.',flush=True)
    def add_scalar(self,tag,value,step):
        self.csv.writerow([step,tag,float(value)]);self.file.flush()
        if self.tb:self.tb.add_scalar(tag,value,step)
    def close(self):
        if self.tb:self.tb.close()
        self.file.close()


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plan','smoke','train','evaluate','view'],default='plan')
    p.add_argument('--reference-results',type=Path,default=ROOT/'data/experiments/l20_five_objects_acquire_10s_20261105')
    p.add_argument('--object',choices=DEFAULTS)
    p.add_argument('--candidate',type=int)
    p.add_argument('--num-envs',type=int,default=12)
    p.add_argument('--iterations',type=int,default=4000,help='Collection iterations; update PPO only when eligible gravity samples exist')
    p.add_argument('--rollout-steps',type=int,default=32)
    p.add_argument('--steps',type=int,default=16,help='Smoke-test policy steps')
    p.add_argument('--episodes',type=int,default=1,help='Complete synchronized six-direction batches per policy/baseline')
    p.add_argument('--view-env',type=int,default=5,help='Camera environment index; 0..5 correspond to gx+,gx-,gy+,gy-,gz+,gz-')
    p.add_argument('--residual-scale',type=float,default=.15)
    p.add_argument('--residual-rate',type=float,default=.2)
    p.add_argument('--position-noise',type=float,default=.0005)
    p.add_argument('--close-seconds',type=float,default=2.)
    p.add_argument('--settle-seconds',type=float,default=1.)
    p.add_argument('--gravity-seconds',type=float,default=10.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import math
    if a.num_envs<6 or a.num_envs%6 or min(a.iterations,a.rollout_steps,a.steps,a.episodes)<1:p.error('Use a positive multiple of six environments and positive lengths')
    if not 0<=a.view_env<a.num_envs:p.error('--view-env must be within num-envs')
    values=[a.residual_scale,a.residual_rate,a.position_noise,a.close_seconds,a.settle_seconds,a.gravity_seconds]
    if not all(math.isfinite(v) for v in values) or min(a.residual_scale,a.residual_rate,a.close_seconds,a.gravity_seconds)<=0 or min(a.position_noise,a.settle_seconds)<0:p.error('Invalid control or phase settings')
    if a.mode in ['evaluate','view'] and not a.checkpoint:p.error('Evaluation/view requires --checkpoint')
    output=a.output.resolve()
    if output.exists() and a.mode!='plan':p.error('Output must be new')
    ckpt=None
    if a.checkpoint:
        ref_saved=json.loads((a.checkpoint.parent/'reference_snapshot.json').read_text())
        ref=load_reference(ref_saved['source_results'],ref_saved['object'],ref_saved['candidate'])
        if ref['scene_sha256']!=ref_saved['scene_sha256'] or ref['targets']!=ref_saved['targets']:p.error('Reference changed since training')
        if a.object is not None and a.object!=ref['object']:p.error('Checkpoint object differs')
        if a.candidate is not None and a.candidate!=ref['candidate']:p.error('Checkpoint candidate differs')
        config=json.loads((a.checkpoint.parent/'config.json').read_text())
        if config['version']!='isaacsim_sdf_residual_v2':p.error('v2 requires a new policy; v1 checkpoints remain available through the v1 entrypoint')
        if config.get('implementation_sha256') is not None and config['implementation_sha256']!=implementation_hashes():
            p.error('RL implementation changed since checkpoint; do not silently resume across code versions')
        config['num_envs']=a.num_envs
    else:
        obj=a.object or 'mug';index=a.candidate if a.candidate is not None else DEFAULTS[obj]
        ref=load_reference(a.reference_results,obj,index)
        config=dict(version='isaacsim_sdf_residual_v2',num_envs=a.num_envs,seed=a.seed,residual_scale=a.residual_scale,
            residual_rate=a.residual_rate,position_noise=a.position_noise,close_seconds=a.close_seconds,
            settle_seconds=a.settle_seconds,gravity_seconds=a.gravity_seconds,open_fraction=.6,physics_dt=1/240,
            decimation=4,kp=5.,kd=.05,collision='SDF256',mimic='physical',algorithm='clipped_PPO',learning_rate=3e-4)
        config['implementation_sha256']=implementation_hashes()
        config.update(learning_phase='gravity_only',strict_history_observations=True,finite_horizon_terminal=True,
                      actor_critic_exclude_warmup_and_absorbing_states=True)
    from isaacsim_grasp.history import EpisodeHistory,FEATURE_NAMES
    try:
        history=EpisodeHistory(a.num_envs,round(config['close_seconds']*240),round(config['settle_seconds']*240),round(config['gravity_seconds']*240))
        if any(s%4 for s in [history.close,history.start,history.total]):p.error('v2 phase boundaries must align with 1/60-second control ticks')
    except ValueError as exc:p.error(str(exc))
    config['history_feature_names']=FEATURE_NAMES
    if a.mode in ['evaluate','view']:config['position_noise']=0.
    plan=dict(mode=a.mode,reference=ref,config=config,iterations=a.iterations,rollout_steps=a.rollout_steps,
        policy='q_target(t) = URDF/mimic_project(q_reference(t) + rate_limited_bounded_residual)',
        action_dim=16,source_candidate_success=ref['reference_is_successful'],
        note='v2: original close/settle reference; learn only eligible gravity transitions; strict failures are absorbing. One reference per policy.')
    if a.mode=='plan':print(json.dumps(plan,indent=2));return
    output.mkdir(parents=True)
    shutil.copy2(ref['scene'],output/'reference_scene.usda')
    ref=dict(ref,scene=str(output/'reference_scene.usda'))
    save(output/'reference_snapshot.json',ref);save(output/'config.json',config);save(output/'plan.json',plan)
    status=lambda **kwargs:save(output/'status.json',kwargs)
    status(status='starting')
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=a.mode!='view',multi_gpu=False,sync_loads=False,create_new_stage=False,
        limit_cpu_threads=8,anti_aliasing=2))
    env=None;writer=None
    try:
        import numpy as np
        import torch
        from isaacsim_grasp.environment_v2 import ResidualGraspEnv
        from isaacsim_grasp.ppo import ActorCritic,advantages,update
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for SDF GPU physics/PPO')
        torch.manual_seed(config['seed']);np.random.seed(config['seed']);device='cuda:0'
        env=ResidualGraspEnv(ref,config,app)
        obs=torch.as_tensor(env.observe(),device=device)
        model=ActorCritic(obs.shape[1],16).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        start_iteration=0
        if a.checkpoint:
            ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
            if ckpt['version']!=config['version'] or ckpt['scene_sha256']!=ref['scene_sha256']:raise ValueError('Checkpoint identity mismatch')
            model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer']);start_iteration=ckpt['iteration']
        save(output/'physics_audit.json',env.audit)
        if a.mode=='smoke':
            for _ in range(a.steps):
                # Deliberately nonzero warmup commands must have no effect.
                action=np.full((env.n,16),3. if env.physics_step<env.gravity_start else 0.,np.float32)
                next_obs,reward,done,info=env.step(action)
                if not np.isfinite(next_obs).all() or not np.isfinite(reward).all():raise RuntimeError('Nonfinite smoke output')
                if not np.allclose(env.last_target,env.current_reference,atol=1e-6):raise RuntimeError('Zero residual does not reproduce reference')
                if done:env.reset()
            save(output/'physics_audit.json',env.audit)
            save(output/'smoke.json',dict(steps=a.steps,environments=env.n,warmup_nonzero_action_ignored=True,zero_gravity_action_reference=True,observation_dim=obs.shape[1],audit=env.audit))
        elif a.mode=='train':
            writer=Metrics(output);episodes=0;optimizer_updates=ckpt.get('optimizer_updates',0) if ckpt else 0;start=time.monotonic()
            for iteration in range(start_iteration+1,start_iteration+a.iterations+1):
                data={k:[] for k in ['obs','actions','logp','values','rewards','next_values','done','terminated','learning_mask']}
                for _ in range(a.rollout_steps):
                    eligible=torch.as_tensor(env.learning_mask(),device=device)
                    with torch.no_grad():
                        dist=model.distribution(obs);action=torch.where(eligible[:,None],dist.sample(),0.);value=model.value(obs);logp=dist.log_prob(action).sum(-1)
                    next_obs,reward,done,info=env.step(action.cpu().numpy())
                    next_tensor=torch.as_tensor(next_obs,device=device)
                    with torch.no_grad():nv=model.value(next_tensor)
                    terminal=torch.as_tensor(info['terminal'],device=device,dtype=torch.bool)
                    for key,val in dict(obs=obs,actions=action,logp=logp,values=value,rewards=torch.as_tensor(reward,device=device),
                        next_values=nv,done=terminal,terminated=terminal,learning_mask=eligible).items():data[key].append(val)
                    if done:
                        episodes+=env.n;writer.add_scalar('evaluation/episode_six_direction_trial_pass_rate',np.mean(info['passed']),iteration)
                        writer.add_scalar('evaluation/all_six_directions_pass',float(all(info['passed'])),iteration)
                        save(output/'last_training_episode.json',info);next_tensor=torch.as_tensor(env.reset(),device=device)
                    obs=next_tensor
                data={k:torch.stack(v) for k,v in data.items()}
                adv,returns=advantages(data['rewards'],data['values'],data['next_values'],data['done'],data['terminated'])
                mask=data['learning_mask'].flatten()
                rollout=[v[mask] for v in [data['obs'].flatten(0,1),data['actions'].flatten(0,1),data['logp'].flatten(),data['values'].flatten(),returns.flatten(),adv.flatten()]]
                loss=vloss=entropy=0.
                if mask.any():
                    loss,vloss,entropy=update(model,optimizer,rollout);optimizer_updates+=1
                writer.add_scalar('train/eligible_gravity_samples',int(mask.sum().item()),iteration)
                writer.add_scalar('train/optimizer_updates',optimizer_updates,iteration)
                writer.add_scalar('train/update_performed',float(mask.any()),iteration)
                writer.add_scalar('train/mean_reward',data['rewards'].mean().item(),iteration)
                if mask.any():
                    writer.add_scalar('train/value_loss',vloss,iteration);writer.add_scalar('train/entropy',entropy,iteration)
                    writer.add_scalar('train/gravity_reward',data['rewards'].flatten()[mask].mean().item(),iteration)
                writer.add_scalar('train/action_std',model.log_std.exp().mean().item(),iteration)
                if iteration%10==0 or iteration==start_iteration+1:
                    value_text=f'{vloss:.3f}' if bool(mask.any()) else 'skipped'
                    print(f'Iteration {iteration}/{start_iteration+a.iterations}: reward={data["rewards"].mean().item():.3f}, value_loss={value_text}, std={model.log_std.exp().mean().item():.3f}, gravity_samples={int(mask.sum())}, optimizer_updates={optimizer_updates}, episodes={episodes}, elapsed={time.monotonic()-start:.1f}s',flush=True)
                    status(status='training',iteration=iteration,episodes=episodes)
                if iteration%100==0 or iteration==start_iteration+a.iterations:
                    torch.save(dict(version=config['version'],scene_sha256=ref['scene_sha256'],iteration=iteration,
                        model=model.state_dict(),optimizer=optimizer.state_dict(),optimizer_updates=optimizer_updates,obs_dim=obs.shape[1],action_dim=16),output/f'model_{iteration:06d}.pt')
            save(output/'physics_audit.json',env.audit)
        else:
            # Exact same reset seed and zero perturbation for a paired baseline.
            summary=dict(reference=dict(object=ref['object'],candidate=ref['candidate']),checkpoint=str(a.checkpoint.resolve()),
                episodes_per_mode=a.episodes,zero_initial_noise=True,runs={})
            for mode in ['zero_residual','policy']:
                env.rng=np.random.default_rng(config['seed']);obs=torch.as_tensor(env.reset(),device=device);trials=[];frames=[]
                for episode in range(a.episodes):
                    while True:
                        started=time.monotonic()
                        with torch.no_grad():action=torch.zeros((env.n,16),device=device) if mode=='zero_residual' else model.actor(obs)
                        next_obs,reward,done,info=env.step(action.cpu().numpy());obs=torch.as_tensor(next_obs,device=device)
                        if episode==0:
                            pos,quat,q,qd,linear,angular=env.state()
                            frames.append(dict(time=env.physics_step*env.dt,phase=0 if env.physics_step<env.close_steps else 1 if env.physics_step<env.gravity_start else 2,
                                object_position=pos.copy(),object_quaternion=quat.copy(),joint_positions=q.copy(),target=env.last_target.copy(),reward=reward.copy()))
                        if a.mode=='view':
                            from isaacsim.core.utils.viewports import set_camera_view
                            if env.physics_step==env.decimation:
                                center=env.origins[a.view_env]+env.rot[a.view_env]@np.array([0.,0.,.04])
                                set_camera_view(eye=center+np.array([.35,.35,.3]),target=center)
                            env.render();remaining=env.dt*env.decimation-(time.monotonic()-started)
                            if remaining>0:time.sleep(remaining)
                        if done:
                            trials.append(info);break
                    if episode+1<a.episodes:obs=torch.as_tensor(env.reset(),device=device)
                per_case={case:[bool(t['passed'][i]) for t in trials for i,c in enumerate(t['cases']) if c==case] for case in ['gx+','gx-','gy+','gy-','gz+','gz-']}
                summary['runs'][mode]=dict(trials=trials,per_direction_pass_rate={c:float(np.mean(v)) for c,v in per_case.items()},
                    all_six_directions_pass=all(all(v) for v in per_case.values()))
                np.savez_compressed(output/f'{mode}_trajectory.npz',time=[f['time'] for f in frames],phase=[f['phase'] for f in frames],
                    object_position=[f['object_position'] for f in frames],object_quaternion=[f['object_quaternion'] for f in frames],
                    joint_positions=[f['joint_positions'] for f in frames],target=[f['target'] for f in frames],dof_names=env.names,cases=env.case_ids)
                save(output/'evaluation.json',summary)
                print(mode,summary['runs'][mode]['per_direction_pass_rate'],flush=True)
            lines=['# Isaac Sim 残差 PPO 评估','',f"参考：{ref['object']} 原始下标 {ref['candidate']}，scale={ref['scale']}。",'',
                '|控制|gx+|gx-|gy+|gy-|gz+|gz-|全部试验通过|','|---|---:|---:|---:|---:|---:|---:|---|']
            for name,run in summary['runs'].items():lines.append('|'+name+'|'+ '|'.join(f'{v:.1%}' for v in run['per_direction_pass_rate'].values())+'|'+str(run['all_six_directions_pass'])+'|')
            lines+=['','对照使用相同参考和确定性初态；这是当前 RL 克隆场景的配对评估，不替代原独立验证器。轨迹只保存首批回合。']
            (output/'REPORT.md').write_text('\n'.join(lines)+'\n');save(output/'physics_audit.json',env.audit)
        status(status='complete',mode=a.mode)
        print(f'Complete: {output}',flush=True)
    except BaseException as exc:
        import traceback
        save(output/'error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        status(status='failed');raise
    finally:
        if writer:writer.close()
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
