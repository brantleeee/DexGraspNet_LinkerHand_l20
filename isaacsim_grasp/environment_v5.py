"""v5: permit bounded residuals in the settle tail, keeping v4 rewards/acceptance."""
import numpy as np
from .environment import CASES
from .environment_v4 import ResidualGraspEnv as V4Env, nonnegative_margin
from .margin_reward import margin_terms
from .preload import action_start_step, eligible


class ResidualGraspEnv(V4Env):
    def __init__(self,reference,config,app):
        self.action_start=action_start_step(round(config['close_seconds']*240),
            round(config['settle_seconds']*240),config['preload_seconds'])
        super().__init__(reference,config,app)
        self.audit.update(task_version='v5',learning_phase='settle_tail_and_gravity',
            preload_seconds=config['preload_seconds'],action_start_seconds=self.action_start*self.dt,
            live_reward_before_gravity=False,acceptance='unchanged_v4_strict_history')

    def learning_mask(self):
        return eligible(self.physics_step,self.action_start,self.total_steps,self.history.failed).copy()

    def step(self,action):
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=(self.n,16) or not np.isfinite(action).all():raise ValueError('Invalid action')
        previous_offset=self.controller.offset.copy()
        failed_before=self.history.failed.copy();learn_before=self.learning_mask()
        effective=np.where(learn_before[:,None],action,0.).astype(np.float32)
        previous=self.previous_action.copy();warmup_error=0.
        for _ in range(self.decimation):
            k=self.physics_step;u=min(1.,(k+1)/self.close_steps);blend=10*u**3-15*u**4+6*u**5
            self.current_reference=np.tile(self.opened+(self.closed-self.opened)*blend,(self.n,1))
            if k<self.action_start:
                # Discard policy output AND filter memory before the settle-tail action window.
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
        if warmup_error>1e-6:raise RuntimeError('v5 protected prefix deviated from original reference')
        self.audit['max_warmup_reference_error_rad']=max(self.audit.get('max_warmup_reference_error_rad',0.),warmup_error)
        self.max_slip=self.history.max_slip.copy();self.max_angle=self.history.max_angle.copy();self.max_depth=self.history.max_depth.copy()
        self.contact_steps=self.history.contact_steps.copy();self.dropped=self.history.dropped.copy()
        requested=self.cfg['residual_scale']*np.tanh(effective)
        terms=margin_terms(np.linalg.norm(pos-self.ref_pos,axis=1),
            np.degrees(2*np.arccos(np.clip(np.abs((quat*self.ref_quat).sum(1)),0,1))),
            np.linalg.norm(linear,axis=1),np.linalg.norm(angular,axis=1),self.force,self.depth,
            self.controller.offset,requested,previous_offset,qd,self.max_slip,self.max_angle,
            self.dt*self.decimation,self.cfg['residual_rate'],self.cfg.get('margin_reward'))
        quality=nonnegative_margin(terms['positive'],terms['costs'])
        # Exactly v4 reward timing: no new live reward in zero gravity. The new
        # preload actions learn via subsequent gravity returns (and prep failures).
        reward=self.history.reward(quality,np.zeros_like(quality),failed_before)
        self.last_reward_terms=dict(terms,unshaped_positive=terms['positive'],shaping_cost=terms['costs'],
                                    positive=quality,costs=np.zeros_like(quality))
        self.last_action_diagnostics=dict(raw_action=action.copy(),requested_residual=requested.copy(),
            applied_residual=self.controller.offset.copy(),joint_position=q.copy(),joint_velocity=qd.copy(),eligible=learn_before)
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
