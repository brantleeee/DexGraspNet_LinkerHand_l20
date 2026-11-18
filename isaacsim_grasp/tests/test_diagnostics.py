import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from isaacsim_grasp.diagnostics import PhysicsDiagnostics


def fake_env():
    n,k=2,3
    q=np.array([[.1,.2,.3],[.1,.2,.3]],np.float32);qd=np.full((n,k),.1,np.float32)
    hand=SimpleNamespace(get_gains=lambda:(np.array([[5.,5.,0.]]*n),np.array([[.05,.05,0.]]*n)),
                         get_measured_joint_efforts=lambda:np.full((n,k),.7,np.float32))
    env=SimpleNamespace(n=n,names=['a','b','c'],ref={'joints':[dict(name=x,limit={'effort':1.}) for x in ['a','b','c']]},
        hand=hand,contact=lambda:None,physics_step=0,dt=1/240,last_target=q+.4,current_reference=q.copy(),
        force=np.ones(n),depth=np.zeros(n),case_ids=np.arange(n),close_steps=480,gravity_start=720,
        ref_pos=np.zeros((n,3)),ref_quat=np.tile([1.,0,0,0],(n,1)),controller=SimpleNamespace(independent=[0,1]))
    env.state=lambda:(np.zeros((n,3)),np.tile([1.,0,0,0],(n,1)),q,qd,np.zeros((n,3)),np.zeros((n,3)))
    return env


def test_every_physics_step_recorded_without_actuation_and_torque_semantics(tmp_path):
    env=fake_env();d=PhysicsDiagnostics(env);d.begin()
    for step in range(1,9):env.physics_step=step;env.contact()
    result=d.finish(tmp_path/'trace.npz')
    with np.load(tmp_path/'trace.npz') as z:
        assert len(z['time'])==9 and result['sample_hz']==240
        assert z['joint_position'].shape==(9,2,3)
        assert np.isnan(z['joint_measured_effort'][0]).all()
        np.testing.assert_allclose(z['joint_measured_effort'][1:],.7)
        np.testing.assert_allclose(z['joint_pd_estimate'][:,:,0:2],1.)
        assert np.all(z['joint_pd_estimate'][:,:,2]==0)
        assert 'not the implicit PhysX drive force' in str(z['torque_note'])
        np.testing.assert_allclose(z['joint_velocity'],.1)


def test_native_smoke_can_export_single_unified_offline_figure(tmp_path):
    # Reuse actual native sensor data only when this optional local artifact exists.
    import pytest,shutil
    source=Path('data/experiments/residual_v2_diagnostics_smoke_20261105/diagnostics_smoke.npz')
    if not source.exists():pytest.skip('Native diagnostic smoke artifact not present')
    from scripts.plot_isaacsim_paired_diagnostics import export,trial_payload
    (tmp_path/'diagnostics').mkdir()
    for mode in ['zero_residual','policy']:shutil.copy2(source,tmp_path/'diagnostics'/f'{mode}.npz')
    summary=dict(diagnostics={mode:[f'diagnostics/{mode}.npz'] for mode in ['zero_residual','policy']},
                 runs={'zero_residual':[{'offset_m':[0.,0.,0.]} for _ in range(6)]})
    (tmp_path/'evaluation.json').write_text(json.dumps(summary))
    destination=export(tmp_path);page=destination.read_text()
    assert 'cdn.plot.ly' not in page.split('<script>')[0]
    assert len(list((tmp_path/'diagnostics_data').glob('*.js')))==6
    assert 'joint_measured_effort' not in page  # raw arrays are in lazy sidecars
    payload=json.loads((tmp_path/'diagnostics_data/group_000_gxplus.js').read_text()[len('window.l20TrialData='):-1])
    assert payload['policy']['measured'][0]==[None]*21
    with np.load(source) as z:
        p=trial_payload(z,0)
        np.testing.assert_allclose(p['linear'],z['object_linear_velocity'][:,0]*1000)
        assert p['measured'].shape==(9,21)


def test_raw_requested_applied_and_actual_motion_are_separate(tmp_path):
    env=fake_env();env.cfg={'residual_scale':.15};env.gravity_start=0;env.close_steps=0
    env.last_target=env.current_reference+.01
    env.learning_mask=lambda:np.ones(2,bool)
    def step(action):
        for _ in range(4):env.physics_step+=1;env.contact()
    env.step=step
    d=PhysicsDiagnostics(env);d.begin();env.step(np.full((2,2),2.))
    result=d.finish(tmp_path/'action.npz')
    with np.load(tmp_path/'action.npz') as z:
        np.testing.assert_allclose(z['raw_action'][1:],2.)
        np.testing.assert_allclose(z['requested_residual'][1:],.15*np.tanh(2.),atol=1e-7)
        np.testing.assert_allclose(z['applied_residual'][1:],.01,atol=1e-6)
        np.testing.assert_allclose(z['joint_velocity'][1:],.1)
        assert result['action_summary'][0]['applied_residual_abs_bias_rad']>0
