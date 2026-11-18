import numpy as np
from isaacsim_grasp.control import ResidualTargets
from isaacsim_grasp.preload_hold import AppliedOffsetHold


def controller():
    joints=[dict(name='parent',limit=dict(lower=-1.,upper=1.),mimic=None),
        dict(name='child',limit=dict(lower=-.3,upper=.3),mimic=dict(joint='parent',multiplier=2.,offset=0.))]
    c=ResidualTargets(joints,['parent','child']);c.reset(1);return c


def test_applied_offset_hold_and_continuous_release_preserve_mimic_rate():
    c=controller();clock=[0];q=np.zeros((1,2),np.float32);act=np.ones((1,1))*10
    hold=AppliedOffsetHold(c,lambda:clock[0],3,7)
    targets=[]
    for k in range(10):
        clock[0]=k
        targets.append(c.step(q,act if k<3 else -act,.01).copy())
    a=np.asarray(targets)
    assert np.array_equal(a[3:7],np.repeat(a[2:3],4,axis=0))
    assert hold.snapshot is not None
    assert np.allclose(a[:,:,1],2*a[:,:,0])
    assert np.max(abs(np.diff(a,axis=0)))<=.2*.01+1e-7
    assert np.max(abs(a[:,:,1]))<=.3
    assert a[7,0,0]<a[6,0,0]  # resumes towards current request, no reset


def test_before_hold_is_identical_and_holds_applied_not_requested_offset():
    a=controller();b=controller();k=[0];h=AppliedOffsetHold(b,lambda:k[0],2,4)
    q=np.zeros((1,2),np.float32);action=np.ones((1,1))*10
    for i in range(2):
        k[0]=i
        assert np.array_equal(a.step(q,action,.01),b.step(q,action,.01))
    saved=b.offset.copy();k[0]=2;b.step(q,-action,.01)
    assert np.array_equal(saved,b.offset)
    assert np.array_equal(h.snapshot,saved)
    assert float(abs(saved[0,0]))<.01  # not a 0.15rad raw request
