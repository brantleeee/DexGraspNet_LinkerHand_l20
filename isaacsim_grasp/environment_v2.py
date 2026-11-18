"""v2: unchanged physical environment, gravity-only actions and observable strict history."""
import numpy as np
from .environment import ResidualGraspEnv as V1Env,CASES
from .history import EpisodeHistory,FEATURE_NAMES


class ResidualGraspEnv(V1Env):
    def __init__(self,reference,config,app):
        self.history=EpisodeHistory(config['num_envs'],round(config['close_seconds']*240),
            round(config['settle_seconds']*240),round(config['gravity_seconds']*240))
        super().__init__(reference,config,app)
        self.audit.update(task_version='v2',learning_phase='gravity_only',history_features=FEATURE_NAMES,
            finite_horizon_terminal=True,strict_failure_is_absorbing=True,positive_reward_after_strict_failure=False)

    def reset(self):
        self.history.reset()
        return super().reset()

    def observe(self):
        return np.concatenate([super().observe(),self.history.features()],axis=1)

    def learning_mask(self):return self.history.learning_mask().copy()

    def step(self,action):
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=(self.n,16) or not np.isfinite(action).all():raise ValueError('Invalid action')
        failed_before=self.history.failed.copy();learn_before=self.learning_mask()
        effective=np.where(learn_before[:,None],action,0.).astype(np.float32)
        previous=self.previous_action.copy();warmup_error=0.
        for _ in range(self.decimation):
            k=self.physics_step;u=min(1.,(k+1)/self.close_steps);blend=10*u**3-15*u**4+6*u**5
            self.current_reference=np.tile(self.opened+(self.closed-self.opened)*blend,(self.n,1))
            if k<self.gravity_start:
                # Discard policy output AND filter memory throughout close/settle.
                self.controller.reset(self.n);self.last_target=self.controller.project(self.current_reference.copy())
                warmup_error=max(warmup_error,float(np.max(np.abs(self.last_target-self.current_reference))))
            else:
                old_offset=self.controller.offset.copy();freeze=self.history.failed.copy()
                self.last_target=self.controller.step(self.current_reference,effective,self.dt,gate=1.)
                self.controller.offset[freeze]=old_offset[freeze]
                self.last_target[freeze]=self.current_reference[freeze]
                self.last_target[np.ix_(freeze,self.controller.independent)]+=old_offset[freeze]
                self.last_target=self.controller.project(self.last_target)
            self.hand.set_joint_position_targets(self.last_target);self.sim.step(render=False);self.physics_step+=1
            if not self.runtime_checked:self.audit_runtime()
            self.contact();pos,quat,q,qd,linear,angular=self.state()
            self.history.record(self.physics_step,self.depth,self.count,self.force,pos,quat)
            if self.physics_step==self.gravity_start:
                self.ref_pos=self.history.ref_pos.copy();self.ref_quat=self.history.ref_quat.copy()
                self.acquired=self.history.acquired.copy();self.scene.CreateGravityMagnitudeAttr(9.81)
            for c,p,m,b in self.controller.mimics:self.max_mimic_error=np.maximum(self.max_mimic_error,np.abs(q[:,c]-m*q[:,p]-b))
        if warmup_error>1e-6:raise RuntimeError('v2 warmup deviated from original reference')
        self.audit['max_warmup_reference_error_rad']=max(self.audit.get('max_warmup_reference_error_rad',0.),warmup_error)
        self.max_slip=self.history.max_slip.copy();self.max_angle=self.history.max_angle.copy();self.max_depth=self.history.max_depth.copy()
        self.contact_steps=self.history.contact_steps.copy();self.dropped=self.history.dropped.copy()
        position=np.linalg.norm(pos-self.ref_pos,axis=1)
        angle=np.degrees(2*np.arccos(np.clip(np.abs((quat*self.ref_quat).sum(1)),0,1)))
        speed=np.linalg.norm(linear,axis=1);spin=np.linalg.norm(angular,axis=1)
        good=(position<=.005)&(angle<=15)&(self.depth<=.001)&(self.force>.001)&(speed<.02)&(spin<.3)
        pose=np.exp(-np.minimum((position/.01)**2+(angle/20)**2,40))
        positive=(1.5*pose+.5*pose*(self.force>.001)+2.*good)*np.exp(-np.minimum((speed/.04)**2+(spin/.6)**2,40))
        costs=.15*np.mean((self.controller.offset/self.cfg['residual_scale'])**2,axis=1)
        costs+=.08*np.mean((np.tanh(effective)-np.tanh(previous))**2,axis=1)
        costs+=.03*np.mean(np.minimum(qd**2,25),axis=1)+2*np.minimum(self.depth/.001,5)
        reward=self.history.reward(positive,costs,failed_before)
        self.previous_action=effective.copy();done=self.physics_step>=self.total_steps
        info=dict(learning_mask=learn_before.tolist(),terminal=self.history.terminal().tolist(),
            strict_failed=self.history.failed.tolist(),new_failure=(self.history.failed&~failed_before).tolist())
        if done:
            info.update(passed=self.history.passed().tolist(),cases=[CASES[c] for c in self.case_ids],
                max_slip_m=self.max_slip.tolist(),max_angle_deg=self.max_angle.tolist(),max_penetration_m=self.max_depth.tolist(),
                acquired=self.acquired.tolist(),max_mimic_error_rad=self.max_mimic_error.tolist(),dropped=self.dropped.tolist(),
                last_second_contact=(self.contact_steps>=240).tolist(),
                failure_reasons={name:self.history.reasons[:,i].tolist() for i,name in enumerate(['slip','angle','penetration','acquisition','last_second_contact'])})
        return self.observe(),reward,done,info
