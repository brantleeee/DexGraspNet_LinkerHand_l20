import numpy as np
from isaacsim_grasp.environment_v4 import nonnegative_margin,ResidualGraspEnv
from isaacsim_grasp.environment_v3 import ResidualGraspEnv as V3Env
from isaacsim_grasp.tests.test_history_v2 import fake_v2_env


def test_nonnegative_quality_removes_negative_survival_counterexample():
    positive=np.array([.1]);cost=np.array([.3]);old=float((positive-cost)[0]);new=float(nonnegative_margin(positive,cost)[0])
    assert old<0 and new>=0
    discount=.99**np.arange(50)
    assert np.sum(discount*old)<-5
    assert np.sum(discount*new)+5*.99**50>0
    assert nonnegative_margin(np.array([1.]),np.array([2.]))[0]<nonnegative_margin(np.array([1.]),np.array([0.]))[0]


def test_v4_preserves_v3_controls_observation_and_strict_outcomes():
    a=fake_v2_env();a.__class__=V3Env;a.cfg['residual_rate']=.2
    b=fake_v2_env();b.__class__=ResidualGraspEnv;b.cfg['residual_rate']=.2
    for k in range(4):
        action=np.ones((6,16))*.5
        oa,ra,da,ia=a.step(action);ob,rb,db,ib=b.step(action)
        np.testing.assert_array_equal(oa,ob);np.testing.assert_array_equal(a.last_target,b.last_target)
        assert ia==ib and da==db
        assert np.all(rb>=0)


def test_v4_failure_penalty_and_absorbing_zero_unchanged():
    e=fake_v2_env();e.__class__=ResidualGraspEnv;e.cfg['residual_rate']=.2;e.slip_once=True
    e.step(np.ones((6,16)));e.step(np.ones((6,16)))
    _,r,_,i=e.step(np.ones((6,16)));assert r[0]==-5 and i['terminal'][0]
    _,r,d,i=e.step(np.zeros((6,16)));assert d and r[0]==0 and not i['passed'][0]
