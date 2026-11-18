import json
from pathlib import Path
import numpy as np
import torch
from isaacsim_grasp.control import ResidualTargets
from isaacsim_grasp.ppo import ActorCritic,advantages,update

ROOT=Path(__file__).resolve().parents[2]


def reference():
    return json.loads((ROOT/'data/experiments/l20_all_acquire_10s_20261104_173323/mug/candidate_020/summary.json').read_text())['manifest']


def test_zero_reference_limit_mimic_and_residual_rate():
    m=reference();joints=[j for j in m['joints'] if j['type']!='fixed'];names=[j['name'] for j in joints]
    c=next(c for c in m['candidates'] if c['index']==20)
    controller=ResidualTargets(joints,names,.15,.2);q=np.tile([c['targets'][n] for n in names],(6,1)).astype(np.float32)
    controller.reset(6)
    np.testing.assert_allclose(controller.step(q,np.zeros((6,16)),1/60),q,atol=1e-6)
    previous=np.zeros((6,16))
    for _ in range(50):
        target=controller.step(q,np.ones((6,16))*100,1/60)
        assert np.max(np.abs(controller.offset))<=.150001
        assert np.max(np.abs(controller.offset-previous))<=.2/60+1e-6
        previous=controller.offset.copy()
        assert np.all(target>=controller.lower-1e-6) and np.all(target<=controller.upper+1e-6)
        for child,parent,mult,offset in controller.mimics:np.testing.assert_allclose(target[:,child],mult*target[:,parent]+offset,atol=1e-6)


def test_timeout_bootstrap_and_terminal_mask():
    rewards=torch.tensor([[1.,1.]]);values=torch.zeros_like(rewards);next_values=torch.tensor([[2.,2.]])
    done=torch.tensor([[True,True]]);terminated=torch.tensor([[False,True]])
    adv,returns=advantages(rewards,values,next_values,done,terminated)
    torch.testing.assert_close(returns,torch.tensor([[2.98,1.]]))


def test_ppo_update_finite_and_initial_actor_zero():
    torch.manual_seed(0);model=ActorCritic(12,16);obs=torch.randn(32,12)
    torch.testing.assert_close(model.actor(obs),torch.zeros(32,16))
    with torch.no_grad():
        dist=model.distribution(obs);action=dist.sample();oldlog=dist.log_prob(action).sum(-1);value=model.value(obs)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    before=model.actor[-1].weight.detach().clone()
    metrics=update(model,optimizer,[obs,action,oldlog,value,torch.randn(32),torch.randn(32)],epochs=2)
    assert np.isfinite(metrics).all()
    assert not torch.equal(before,model.actor[-1].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
