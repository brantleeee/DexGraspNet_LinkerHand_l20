"""Read-only 240 Hz position/velocity/effort recording for paired evaluation."""
from pathlib import Path
import json
import numpy as np

TORQUE_NOTE=('joint_measured_effort is Isaac Sim get_measured_joint_efforts: projected joint effort, '
             'which can include constraint reactions; it is not an isolated motor-output measurement. '
             'joint_pd_estimate is clipped Kp*(target-q)-Kd*qdot using the sampled state, '
             'not the implicit PhysX drive force or measured motor torque.')


class PhysicsDiagnostics:
    def __init__(self,env):
        self.env=env;self.active=False;self.frames=[]
        self.original_contact=env.contact
        self.raw_action=np.zeros((env.n,len(env.controller.independent)),np.float32)
        self.action_eligible=np.zeros(env.n,bool)
        if hasattr(env,'step'):
            original_step=env.step
            def step(action):
                self.raw_action=np.asarray(action,dtype=np.float32).copy()
                self.action_eligible=np.asarray(env.learning_mask(),dtype=bool).copy()
                return original_step(action)
            env.step=step
        # Existing v2 calls contact once after each physics step and before its
        # history update. Add only sensor reads; no changes to controls/physics.
        def contact():
            self.original_contact()
            if self.active:self.capture()
        env.contact=contact
        joints={j['name']:j for j in env.ref['joints']}
        self.effort_limits=np.array([joints[n]['limit']['effort'] for n in env.names])
        self.kp,self.kd=[np.asarray(v).copy() for v in env.hand.get_gains()]

    def begin(self):
        self.frames=[];self.active=True
        self.raw_action[:]=0;self.action_eligible[:]=False
        self.capture(initial=True)

    def capture(self,initial=False):
        env=self.env;pos,quat,q,qd,linear,angular=env.state()
        measured=np.full(q.shape,np.nan,dtype=np.float32) if initial else env.hand.get_measured_joint_efforts()
        if measured is None:raise RuntimeError('Measured joint effort unavailable; refusing to label an estimate as measured')
        measured=np.asarray(measured)
        if measured.shape!=q.shape or (not initial and not np.isfinite(measured).all()):raise RuntimeError('Invalid measured joint effort')
        estimate=np.clip(self.kp*(env.last_target-q)-self.kd*qd,-self.effort_limits,self.effort_limits)
        frame=dict(time=env.physics_step*env.dt,
            object_position=pos,object_quaternion=quat,object_linear_velocity=linear,object_angular_velocity=angular,
            joint_position=q,joint_velocity=qd,joint_target=env.last_target,joint_reference=env.current_reference,
            joint_measured_effort=measured,joint_pd_estimate=estimate,contact_force=env.force,penetration=env.depth)
        ids=env.controller.independent
        amplitude=env.cfg['residual_scale'] if hasattr(env,'cfg') else 0.
        requested=amplitude*np.tanh(self.raw_action)
        eligible=self.action_eligible&(~env.history.failed if hasattr(env,'history') else True)
        frame.update(raw_action=self.raw_action,
            requested_residual=requested,effective_requested_residual=np.where(eligible[:,None],requested,0.),
            applied_residual=(env.last_target-env.current_reference)[:,ids],action_enabled=eligible,
            phase=np.full(env.n,0 if env.physics_step<env.close_steps else 1 if env.physics_step<env.gravity_start else 2))
        for key,value in frame.items():
            if key not in ['time','joint_measured_effort'] and not np.isfinite(value).all():raise RuntimeError(f'Nonfinite diagnostic: {key}')
        self.frames.append({key:np.array(value,copy=True) for key,value in frame.items()})

    def finish(self,path):
        self.active=False
        if len(self.frames)<2:raise ValueError('No recorded physics steps')
        data={key:np.asarray([frame[key] for frame in self.frames]) for key in self.frames[0]}
        env=self.env
        if not np.allclose(np.diff(data['time']),env.dt,rtol=0,atol=1e-8):raise RuntimeError('Diagnostic samples skipped physics steps')
        data.update(dof_names=np.asarray(env.names),independent_ids=np.asarray(env.controller.independent),
            effort_limits=self.effort_limits,cases=env.case_ids,
            gravity_reference_position=env.ref_pos,gravity_reference_quaternion=env.ref_quat,
            initial_object_position=data['object_position'][0],close_seconds=env.close_steps*env.dt,
            gravity_start_seconds=env.gravity_start*env.dt,dt=env.dt,torque_note=TORQUE_NOTE)
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**data)
        from .action_analysis import action_summaries
        action_stats=action_summaries(data)
        self.frames=[]
        return dict(file=str(path),frames=len(data['time']),sample_hz=1/env.dt,torque_note=TORQUE_NOTE,action_summary=action_stats)
