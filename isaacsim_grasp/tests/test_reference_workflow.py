import numpy as np
from isaacsim_grasp.history import EpisodeHistory
from isaacsim_grasp.reference_workflow import FirstFailureRecorder,admission


def advance(h,x=0.,depth=0.,count=1):
    h.record(h.step+1,np.full(h.n,depth),np.full(h.n,count),np.ones(h.n),np.tile([x,0,0],(h.n,1)),np.tile([1.,0,0,0],(h.n,1)))


def test_first_gravity_failure_admits_and_is_not_overwritten():
    h=EpisodeHistory(1,4,4,8,.25);r=FirstFailureRecorder(h)
    for _ in range(8):advance(h)
    advance(h,x=.006);first=r.first[0].copy()
    assert first['phase']=='gravity_test' and first['physics_step']==9 and first['gravity_seconds']==.25
    while h.step<h.total:advance(h,x=.1)
    assert r.first[0]==first
    assert admission([dict(passed=False,acquired=True,first_failure=first)])['suitable']


def test_warmup_and_acquisition_failures_are_not_gravity_repairable():
    h=EpisodeHistory(1,4,4,8,.25);r=FirstFailureRecorder(h);advance(h,depth=.0011)
    assert r.first[0]['phase']=='close_zero_g'
    assert not admission([dict(passed=False,acquired=True,first_failure=r.first[0])])['suitable']
    h.reset();r.begin()
    while h.step<8:advance(h,count=0)
    assert r.first[0]['phase']=='acquisition_boundary'
    assert not admission([dict(passed=False,acquired=False,first_failure=r.first[0])])['suitable']


def test_all_passed_is_not_a_failed_reference_repair_problem():
    result=admission([dict(passed=True,acquired=True,first_failure=None)])
    assert not result['suitable'] and result['reason']=='no_baseline_failure_to_repair'
