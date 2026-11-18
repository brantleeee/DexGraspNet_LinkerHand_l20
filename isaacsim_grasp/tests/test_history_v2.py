import numpy as np
import torch
from isaacsim_grasp.history import EpisodeHistory,FEATURE_NAMES
from isaacsim_grasp.ppo import advantages


def record(h,*,depth=0.,force=1.,count=1,x=0.):
    h.record(h.step+1,np.full(h.n,depth),np.full(h.n,count),np.full(h.n,force),
             np.tile([x,0.,0.],(h.n,1)),np.tile([1.,0.,0.,0.],(h.n,1)))


def test_history_features_expose_time_and_irrecoverable_slip():
    h=EpisodeHistory(1,4,0,8,dt=.25)
    initial=h.features().copy()
    for _ in range(4):record(h)
    assert h.acquired[0] and h.learning_mask()[0]
    healthy=h.features().copy();before=h.failed.copy();record(h,x=.006)
    assert h.failed[0] and h.terminal()[0] and not h.learning_mask()[0]
    assert h.reward(np.array([4.]),np.array([0.]),before)[0]==-5
    record(h,x=0.)
    assert h.reward(np.array([4.]),np.array([0.]),h.failed.copy())[0]==0
    assert not np.array_equal(h.features(),healthy)
    assert h.features().shape==(1,len(FEATURE_NAMES)) and len(FEATURE_NAMES)==27
    assert h.features()[0,FEATURE_NAMES.index('failed')]==1
    while h.step<h.total:record(h)
    assert not h.passed()[0] and np.isfinite(h.features()).all()
    assert initial[0,0]==0 and h.features()[0,0]==1


def test_contact_gap_before_final_window_can_recover():
    h=EpisodeHistory(1,4,0,8,dt=.25)
    while h.step<8:record(h)
    # At step8 there are still four steps left: a gap is recoverable.
    # Restart and put the gap at step8 explicitly.
    h.reset()
    while h.step<7:record(h)
    record(h,force=0.,count=0)
    assert not h.failed[0]
    while h.step<h.total:record(h)
    assert h.passed()[0] and h.terminal()[0]
    assert h.reward(np.array([4.]),np.array([0.]),np.array([False]))[0]==9


def test_contact_gap_inside_final_window_is_terminal_even_if_recovered():
    h=EpisodeHistory(1,4,0,8,dt=.25)
    while h.step<8:record(h)
    record(h,force=0.,count=0)
    assert h.failed[0] and h.reasons[0,4]
    while h.step<h.total:record(h)
    assert not h.passed()[0]


def test_warmup_depth_and_missing_acquisition_cannot_earn_hold_reward():
    for kind in ['depth','acquisition']:
        h=EpisodeHistory(1,4,0,8,dt=.25)
        for k in range(4):record(h,depth=.0011 if kind=='depth' and k==0 else 0.,count=0 if kind=='acquisition' else 1)
        assert h.failed[0] and not h.learning_mask()[0]
        assert h.reward(np.array([4.]),np.array([0.]),h.failed.copy())[0]==0


def test_finite_horizon_success_does_not_bootstrap():
    h=EpisodeHistory(1,4,0,8,dt=.25)
    while h.step<h.total:record(h)
    assert h.passed()[0] and h.terminal()[0]
    done=torch.as_tensor(h.terminal()).reshape(1,1)
    _,ret=advantages(torch.tensor([[5.]]),torch.tensor([[0.]]),torch.tensor([[100.]]),done,done)
    torch.testing.assert_close(ret,torch.tensor([[5.]]))


def fake_v2_env():
    """Exercise actual v2 stepping/controller with deterministic fake physics."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from isaacsim_grasp.environment_v2 import ResidualGraspEnv
    from isaacsim_grasp.control import ResidualTargets
    root=Path(__file__).resolve().parents[2]
    m=json.loads((root/'data/experiments/l20_all_acquire_10s_20261104_173323/mug/candidate_020/summary.json').read_text())['manifest']
    c=next(c for c in m['candidates'] if c['index']==20)
    e=ResidualGraspEnv.__new__(ResidualGraspEnv);e.n=6;e.dt=.25;e.decimation=4
    e.close_steps=4;e.settle_steps=4;e.gravity_start=8;e.gravity_steps=8;e.total_steps=16;e.physics_step=0
    joints=[j for j in m['joints'] if j['type']!='fixed'];e.names=[j['name'] for j in joints]
    e.controller=ResidualTargets(joints,e.names,.15,.2);e.controller.reset(e.n)
    e.closed=np.array([c['targets'][n] for n in e.names],np.float32);e.opened=e.controller.project((e.closed*.5)[None,:])[0]
    e.cfg={'residual_scale':.15};e.rot=np.tile(np.eye(3),(e.n,1,1));e.ref_pos=np.zeros((e.n,3));e.ref_quat=np.tile([1.,0,0,0],(e.n,1))
    e.current_reference=np.tile(e.opened,(e.n,1));e.last_target=e.current_reference.copy();e.previous_action=np.zeros((e.n,16))
    e.max_mimic_error=np.zeros(e.n);e.acquired=np.zeros(e.n,bool);e.force=np.zeros(e.n);e.count=np.zeros(e.n,int);e.depth=np.zeros(e.n)
    e.audit={};e.runtime_checked=True;e.history=EpisodeHistory(e.n,4,4,8,dt=.25);e.case_ids=np.arange(6)
    e.scene=SimpleNamespace(CreateGravityMagnitudeAttr=lambda value:None)
    e.hand=SimpleNamespace(set_joint_position_targets=lambda value:None);e.sim=SimpleNamespace(step=lambda **kwargs:None)
    e.slip_once=False
    def state():
        p=np.zeros((e.n,3))
        if e.slip_once and e.physics_step==9:p[0,0]=.006
        return p,np.tile([1.,0,0,0],(e.n,1)),e.last_target.copy(),np.zeros((e.n,21)),np.zeros((e.n,3)),np.zeros((e.n,3))
    def contact():e.force=np.ones(e.n);e.count=np.ones(e.n,int);e.depth=np.zeros(e.n)
    e.state=state;e.contact=contact
    return e


def test_v2_nonzero_warmup_action_is_ignored_and_gravity_is_enabled():
    e=fake_v2_env();a=np.full((6,16),100.,np.float32)
    e.controller.offset[:]=.1
    for _ in range(2):
        obs,reward,done,info=e.step(a)
        np.testing.assert_allclose(e.last_target,e.current_reference,atol=1e-6)
        assert not np.any(e.controller.offset) and not np.any(e.previous_action)
        assert not np.any(reward) and not any(info['learning_mask'])
        assert obs.shape==(6,120)
    assert e.learning_mask().all()
    obs,reward,done,info=e.step(a)
    assert np.any(np.abs(e.controller.offset)>1e-6) and all(info['learning_mask']) and np.all(reward>0)
    for c,p,m,b in e.controller.mimics:np.testing.assert_allclose(e.last_target[:,c],m*e.last_target[:,p]+b,atol=1e-6)


def test_v2_transient_failure_stays_terminal_freezes_target_and_cannot_reearn():
    e=fake_v2_env();a=np.full((6,16),100.,np.float32)
    e.step(a);e.step(a);e.slip_once=True
    _,reward,done,info=e.step(a)
    assert reward[0]==-5 and info['terminal'][0] and not e.learning_mask()[0]
    frozen=e.last_target[0].copy()
    _,reward,done,info=e.step(-a)
    assert done and not info['passed'][0] and all(info['passed'][1:])
    assert reward[0]==0 and all(info['terminal'])
    np.testing.assert_allclose(e.last_target[0],frozen,atol=1e-6)
