"""Markov history and the single strict success/failure definition for residual v2."""
import numpy as np

REASONS=['slip','angle','penetration','acquisition','last_second_contact']
FEATURE_NAMES=[
    'episode_elapsed','episode_remaining','close_progress','settle_progress','gravity_progress',
    'max_slip_over_limit','max_angle_over_limit','max_penetration_over_limit',
    'current_penetration_over_limit','continuous_contact_fraction','contact_count_log',
    'acquired','acquisition_checked','failed','dropped',
    *['failed_'+r for r in REASONS],
    'gravity_reference_x','gravity_reference_y','gravity_reference_z',
    'gravity_reference_qw','gravity_reference_qx','gravity_reference_qy','gravity_reference_qz',
]


class EpisodeHistory:
    def __init__(self,n,close_steps,settle_steps,gravity_steps,dt=1/240):
        self.n=n;self.close=close_steps;self.settle=settle_steps;self.start=close_steps+settle_steps
        self.gravity=gravity_steps;self.total=self.start+gravity_steps;self.dt=dt;self.tail=round(1/dt)
        if self.close<1 or self.gravity<self.tail:raise ValueError('v2 needs positive close time and at least one gravity second')
        self.reset()

    def reset(self):
        self.step=0;self.max_slip=np.zeros(self.n);self.max_angle=np.zeros(self.n);self.max_depth=np.zeros(self.n)
        self.depth=np.zeros(self.n);self.contact_steps=np.zeros(self.n,int);self.count=np.zeros(self.n,int)
        self.acquired=np.zeros(self.n,bool);self.acquisition_checked=False
        self.failed=np.zeros(self.n,bool);self.dropped=np.zeros(self.n,bool)
        self.reasons=np.zeros((self.n,len(REASONS)),bool)
        self.ref_pos=np.zeros((self.n,3));self.ref_quat=np.tile([1.,0,0,0],(self.n,1))

    def record(self,step,depth,count,force,pos,quat):
        """Called after every physics step, including all zero-gravity steps."""
        if step!=self.step+1:raise ValueError('History must be updated every physics step')
        self.step=step;self.depth=np.asarray(depth).copy();self.count=np.asarray(count).copy()
        self.max_depth=np.maximum(self.max_depth,self.depth)
        self.reasons[:,2]|=self.max_depth>.001
        if step==self.start:
            self.ref_pos=pos.copy();self.ref_quat=quat.copy();self.acquired=self.count>0
            self.acquisition_checked=True;self.reasons[:,3]|=~self.acquired
        if step>self.start:
            slip=np.linalg.norm(pos-self.ref_pos,axis=1)
            angle=np.degrees(2*np.arccos(np.clip(np.abs((quat*self.ref_quat).sum(1)),0,1)))
            self.max_slip=np.maximum(self.max_slip,slip);self.max_angle=np.maximum(self.max_angle,angle)
            self.contact_steps=np.where(np.asarray(force)>.001,self.contact_steps+1,0)
            self.reasons[:,0]|=self.max_slip>.005;self.reasons[:,1]|=self.max_angle>15
            # Contact can recover early in the episode. Fail only when it can no
            # longer fill the complete final one-second window before the deadline.
            self.reasons[:,4]|=self.contact_steps+(self.total-step)<self.tail
            self.dropped|=(slip>.05)|(angle>90)
        self.failed|=self.reasons.any(1)

    def learning_mask(self):
        return (self.step>=self.start)&(self.step<self.total)&(~self.failed)

    def passed(self):
        return (self.step>=self.total)&self.acquired&(~self.failed)&(self.contact_steps>=self.tail)

    def terminal(self):
        # A fixed 13-second grasp task is finite-horizon, including successful
        # endings; do not bootstrap it as an infinite-horizon time-limit truncation.
        return self.failed|(self.step>=self.total)

    def features(self):
        time=[self.step/self.total,(self.total-self.step)/self.total,
              min(self.step/self.close,1.),np.clip((self.step-self.close)/max(self.settle,1),0,1),
              np.clip((self.step-self.start)/self.gravity,0,1)]
        features=np.concatenate([np.tile(time,(self.n,1)),
            self.max_slip[:,None]/.005,self.max_angle[:,None]/15,self.max_depth[:,None]/.001,
            self.depth[:,None]/.001,np.minimum(self.contact_steps/self.tail,1)[:,None],np.log1p(self.count[:,None])/5,
            self.acquired[:,None],np.full((self.n,1),self.acquisition_checked),self.failed[:,None],self.dropped[:,None],
            self.reasons,self.ref_pos/.02,self.ref_quat],axis=1)
        assert features.shape==(self.n,len(FEATURE_NAMES))
        return np.clip(features,-10,10).astype(np.float32)

    def reward(self,positive,costs,failed_before,success_bonus=5.):
        new_failure=self.failed&~failed_before
        active=(self.step>self.start)&~self.failed
        reward=np.where(active,positive-costs,0.)-5.*new_failure
        if self.step==self.total:reward+=success_bonus*self.passed()
        return reward.astype(np.float32)
