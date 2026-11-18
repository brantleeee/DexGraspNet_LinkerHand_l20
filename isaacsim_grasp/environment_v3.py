"""v3 changes reward only; v2 reference, physical dynamics and acceptance stay intact."""
import numpy as np
from .environment_v2 import ResidualGraspEnv as V2Env
from .margin_reward import margin_terms


class ResidualGraspEnv(V2Env):
    def __init__(self,reference,config,app):
        super().__init__(reference,config,app)
        self.audit.update(task_version='v3',objective='continuous_stability_margin_and_small_residual',acceptance='unchanged_v2_strict_history')

    def step(self,action):
        raw=np.asarray(action,dtype=np.float32);previous_offset=self.controller.offset.copy();failed_before=self.history.failed.copy()
        eligible=self.learning_mask().copy()
        obs,_,done,info=super().step(raw)
        pos,quat,q,qd,linear,angular=self.state()
        requested=self.cfg['residual_scale']*np.tanh(np.where(eligible[:,None],raw,0.))
        terms=margin_terms(np.linalg.norm(pos-self.ref_pos,axis=1),
            np.degrees(2*np.arccos(np.clip(np.abs((quat*self.ref_quat).sum(1)),0,1))),
            np.linalg.norm(linear,axis=1),np.linalg.norm(angular,axis=1),self.force,self.depth,
            self.controller.offset,requested,previous_offset,qd,self.max_slip,self.max_angle,self.dt*self.decimation,
            self.cfg['residual_rate'],self.cfg.get('margin_reward'))
        reward=self.history.reward(terms['positive'],terms['costs'],failed_before)
        self.last_reward_terms=terms
        self.last_action_diagnostics=dict(raw_action=raw.copy(),requested_residual=requested.copy(),
            applied_residual=self.controller.offset.copy(),joint_position=q.copy(),joint_velocity=qd.copy(),eligible=eligible)
        return obs,reward,done,info
