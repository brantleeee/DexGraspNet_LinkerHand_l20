"""New live dynamics validation with simultaneous native RGB recording."""
import argparse,json,os,sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.paired_evaluation import FixedUniform,validate_samples


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    spec=json.loads(a.plan.read_text());out=a.output.resolve();out.mkdir(parents=True)
    save(out/'plan.json',spec);save(out/'status.json',dict(status='starting',pid=os.getpid()))
    print(f'[isolated] starting pid={os.getpid()} output={out}',flush=True)
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,multi_gpu=False,sync_loads=False,create_new_stage=False,limit_cpu_threads=8,anti_aliasing=2));env=None
    try:
        import torch
        from isaacsim_grasp.environment_v5 import ResidualGraspEnv
        from isaacsim_grasp.finger_diagnostics import FingerDiagnostics
        from isaacsim_grasp.reference_workflow import FirstFailureRecorder
        from isaacsim_grasp.evaluation_interventions import scale_preload_action
        from isaacsim_grasp.reference import digest
        from evaluate_isaacsim_residual_paired import initial_state
        torch.manual_seed(spec['config']['seed']);np.random.seed(spec['config']['seed'])
        ref=spec['reference'];config=spec['config']
        if digest(ref['scene'])!=ref['scene_sha256']:raise ValueError('Scene changed since plan')
        if spec.get('variant'):
            from isaacsim_grasp.reference_variants import apply_variant
            ref=apply_variant(ref,spec['variant'],out)
        save(out/'reference_snapshot.json',ref);save(out/'config.json',config)
        save(out/'status.json',dict(status='initializing_scene',pid=os.getpid()))
        print('[isolated] initializing SDF scene and physics',flush=True)
        env=ResidualGraspEnv(ref,config,app)
        offsets=validate_samples(spec['samples'],1,6,config['position_noise'])[0]
        # Fresh simulation, no prior episode and no warmup physics steps.
        env.rng=FixedUniform(offsets,config['position_noise']);obs=env.reset()
        state=initial_state(env);np.savez_compressed(out/'initial_states.npz',**state)
        hold=None
        if spec.get('hold_start_seconds') is not None:
            from isaacsim_grasp.preload_hold import AppliedOffsetHold
            start=int(round(spec['hold_start_seconds']/env.dt))
            if abs(start*env.dt-spec['hold_start_seconds'])>1e-8 or start%env.decimation:
                raise ValueError('Hold start must align with policy and physics steps')
            if not env.action_start<start<env.gravity_start:raise ValueError('Hold start must lie in preload window')
            hold=AppliedOffsetHold(env.controller,lambda:env.physics_step,start,env.gravity_start)
        recorder=FingerDiagnostics(env);failure=FirstFailureRecorder(env.history);recorder.begin();model=None
        if spec['controller']=='policy':
            from isaacsim_grasp.ppo import ActorCritic
            ckpt=torch.load(spec['checkpoint'],map_location='cuda:0',weights_only=False)
            if ckpt['version']!=config['version'] or ckpt['scene_sha256']!=ref['scene_sha256']:raise ValueError('Policy/reference mismatch')
            model=ActorCritic(ckpt['obs_dim'],16).to('cuda:0');model.load_state_dict(ckpt['model']);model.eval()
        from isaacsim_grasp.live_video import LiveDynamicsVideo
        video=LiveDynamicsVideo(env,out);video.capture()
        print(f'[isolated] scene ready; simulating {env.total_steps*env.dt:g}s in six directions',flush=True)
        while True:
            raw=np.zeros((env.n,16),np.float32)
            if model is not None:
                with torch.inference_mode():raw=model.actor(torch.as_tensor(obs,device='cuda:0')).cpu().numpy()
            command,gain=scale_preload_action(raw,spec.get('preload_gain',1.),env.physics_step,env.action_start,env.gravity_start)
            obs,reward,done,info=env.step(command)
            if env.physics_step%8==0:video.capture()
            if env.physics_step%240==0 or done:
                elapsed=env.physics_step*env.dt;total=env.total_steps*env.dt
                save(out/'status.json',dict(status='simulating',pid=os.getpid(),physics_step=env.physics_step,
                    simulation_seconds=elapsed,total_simulation_seconds=total,strict_failed=int(env.history.failed.sum())))
                print(f'[isolated] simulation {elapsed:.1f}/{total:.1f}s; strict_failed={int(env.history.failed.sum())}/{env.n}',flush=True)
            if done:break
        video_audit=video.finish()
        print('[isolated] saving 240Hz trajectory and contact points',flush=True)
        save(out/'status.json',dict(status='saving',pid=os.getpid()))
        detail=recorder.finish(out/'trajectory.npz');save(out/'contact_events.json',detail['finger_contact_events'])
        save(out/'physics_audit.json',env.audit)
        result=dict(complete=True,pid=os.getpid(),controller=spec['controller'],reference=dict(object=ref['object'],candidate=ref['candidate'],scale=ref['scale']),
            info=info,first_failure=failure.first,diagnostics=detail,source_group=spec['samples']['source_group'],
            process_contract='one fresh SimulationApp and stage per process; one episode only',
            note='Contact normals retain SDK sign; friction anchors are not paired one-to-one with normal contacts')
        result['video_audit']=video_audit
        result['evaluation_intervention']=dict(type='applied_offset_hold',start_seconds=spec.get('hold_start_seconds'),end_seconds=env.gravity_start*env.dt,training=False,held_offset_rad=hold.snapshot.tolist() if hold is not None and hold.snapshot is not None else None)
        save(out/'result.json',result);save(out/'status.json',dict(status='complete',pid=os.getpid()))
        print(f'Complete: {out}',flush=True)
    except BaseException as exc:
        import traceback
        save(out/'error.json',dict(message=str(exc),traceback=traceback.format_exc()));save(out/'status.json',dict(status='failed',pid=os.getpid()));raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
