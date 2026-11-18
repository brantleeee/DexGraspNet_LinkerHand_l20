"""v4 fixes the negative-survival incentive without changing dynamics or exploration."""
import numpy as np
from .environment_v3 import ResidualGraspEnv as V3Env


def nonnegative_margin(positive,costs):
    positive=np.asarray(positive);costs=np.asarray(costs)
    if not np.isfinite(positive).all() or not np.isfinite(costs).all() or np.any(positive<0) or np.any(costs<0):
        raise ValueError('Quality and shaping costs must be finite and nonnegative')
    return positive/(1.+costs)


class ResidualGraspEnv(V3Env):
    def __init__(self,reference,config,app):
        super().__init__(reference,config,app)
        self.audit.update(task_version='v4',objective='nonnegative_margin_quality_discount',
                          live_reward_nonnegative=True,terminal_failure_penalty_unchanged=True)

    def step(self,action):
        failed_before=self.history.failed.copy()
        obs,_,done,info=super().step(action)
        terms=self.last_reward_terms.copy();quality=nonnegative_margin(terms['positive'],terms['costs'])
        reward=self.history.reward(quality,np.zeros_like(quality),failed_before)
        self.last_reward_terms=dict(terms,unshaped_positive=terms['positive'],shaping_cost=terms['costs'],
                                    positive=quality,costs=np.zeros_like(quality))
        return obs,reward,done,info
