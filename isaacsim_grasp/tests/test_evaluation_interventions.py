import numpy as np
import pytest
from isaacsim_grasp.evaluation_interventions import scale_preload_action,InterventionDiagnostics


def test_half_gain_in_tanh_space_only_and_exact_identity_outside():
    raw=np.array([[-100.,-2.,0.,2.,100.]],np.float32)
    for k in [0,599,720,800]:
        command,gain=scale_preload_action(raw,.5,k,600,720)
        np.testing.assert_array_equal(command,raw);assert gain==1.
    for k in [600,719]:
        command,gain=scale_preload_action(raw,.5,k,600,720)
        assert gain==.5 and np.isfinite(command).all()
        np.testing.assert_allclose(np.tanh(command),.5*np.tanh(raw),atol=1e-7)
    full,_=scale_preload_action(raw,1.,600,600,720)
    np.testing.assert_array_equal(full,raw)
    zero,_=scale_preload_action(raw,0.,600,600,720)
    np.testing.assert_array_equal(zero,np.zeros_like(raw))
    assert raw[0,0]==-100.


def test_invalid_gain_rejected():
    for gain in [-.1,1.1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):scale_preload_action(np.zeros((1,16)),gain,600,600,720)


def test_environment_rate_limit_continuous_across_gravity_boundary():
    from isaacsim_grasp.tests.test_preload_v5 import fake,V5Env
    e=fake(V5Env)
    raw=np.full((6,16),10.,np.float32)
    previous=e.controller.offset.copy();observed=[]
    original=e.hand.set_joint_position_targets
    def target(q):
        nonlocal previous
        offset=e.controller.offset.copy()
        assert np.max(abs(offset-previous))<=e.cfg['residual_rate']*e.dt+1e-6
        previous=offset;observed.append(offset.copy());original(q)
    e.hand.set_joint_position_targets=target
    for _ in range(4):
        cmd,_=scale_preload_action(raw,.5,e.physics_step,e.action_start,e.gravity_start)
        e.step(cmd)
    assert np.max(abs(observed[7]))>0
    assert np.max(abs(observed[8]))>0  # no offset clearing when gravity starts


def test_diagnostics_preserve_actor_and_command_layers(tmp_path):
    from isaacsim_grasp.tests.test_diagnostics import fake_env
    env=fake_env();env.cfg={'residual_scale':.15};env.close_steps=0;env.gravity_start=0
    env.learning_mask=lambda:np.ones(2,bool)
    def step(action):
        for _ in range(4):env.physics_step+=1;env.contact()
    env.step=step
    d=InterventionDiagnostics(env);d.begin()
    raw=np.full((2,2),2.,np.float32)
    cmd,gain=scale_preload_action(raw,.5,600,600,720)
    d.set_policy_action(raw,gain);env.step(cmd);d.finish(tmp_path/'layers.npz')
    with np.load(tmp_path/'layers.npz') as z:
        np.testing.assert_allclose(z['policy_raw_action'][1:],2.)
        np.testing.assert_allclose(z['requested_residual'][1:],.5*z['policy_requested_residual'][1:],atol=1e-7)
        np.testing.assert_allclose(z['preload_request_gain'][1:],.5)
