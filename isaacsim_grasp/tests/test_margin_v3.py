import numpy as np
from isaacsim_grasp.margin_reward import margin_terms
from isaacsim_grasp.tests.test_history_v2 import fake_v2_env
from isaacsim_grasp.environment_v3 import ResidualGraspEnv


def terms(position=0.,angle=0.,offset=0.,requested=0.):
    z=np.zeros(1);a=np.full((1,16),offset);r=np.full((1,16),requested)
    return margin_terms(np.array([position]),np.array([angle]),z,z,np.ones(1),z,a,r,a,np.zeros((1,21)),z,z,1/60,.2)


def test_reward_distinguishes_safe_but_lower_margin_pose_and_unnecessary_bias():
    zero=terms();moved=terms(.0022,3.23);biased=terms(offset=.03,requested=.03)
    assert zero['positive']>moved['positive'] and zero['costs']==0
    assert biased['costs']>0 and biased['positive']==zero['positive']
    assert zero['positive']-zero['costs']>biased['positive']-biased['costs']


def test_v3_changes_reward_without_changing_commands_physics_or_acceptance():
    a=fake_v2_env();b=fake_v2_env();b.__class__=ResidualGraspEnv;b.cfg['residual_rate']=.2
    for step in range(4):
        action=np.full((6,16),0.8 if step>=2 else 10.,np.float32)
        oa,ra,da,ia=a.step(action);ob,rb,db,ib=b.step(action)
        np.testing.assert_array_equal(a.last_target,b.last_target)
        np.testing.assert_array_equal(oa,ob)
        assert da==db and ia==ib
        if step<2:np.testing.assert_array_equal(rb,np.zeros(6))
        else:assert np.all(rb<ra)


def test_v3_cannot_reearn_reward_after_strict_failure():
    e=fake_v2_env();e.__class__=ResidualGraspEnv;e.cfg['residual_rate']=.2;e.slip_once=True
    e.step(np.ones((6,16)));e.step(np.ones((6,16)))
    _,reward,_,info=e.step(np.ones((6,16)))
    assert reward[0]==-5 and info['terminal'][0]
    _,reward,done,info=e.step(np.zeros((6,16)))
    assert done and reward[0]==0 and not info['passed'][0]
