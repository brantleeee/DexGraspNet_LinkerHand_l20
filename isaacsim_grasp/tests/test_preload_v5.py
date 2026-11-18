import numpy as np
import pytest
from isaacsim_grasp.preload import action_start_step, eligible
from isaacsim_grasp.environment_v5 import ResidualGraspEnv as V5Env
from isaacsim_grasp.environment_v4 import ResidualGraspEnv as V4Env
from isaacsim_grasp.tests.test_history_v2 import fake_v2_env


def fake(cls):
    e=fake_v2_env();e.__class__=cls;e.cfg['residual_rate']=.2
    if cls is V5Env:e.action_start=4  # tiny fake clock: all settle is eligible
    return e


def test_real_schedule_boundary_and_failures():
    start=action_start_step(480,240,.5)
    assert start==600
    failed=np.array([False,True])
    assert not eligible(599,start,3120,failed).any()
    assert eligible(600,start,3120,failed).tolist()==[True,False]
    assert not eligible(3120,start,3120,failed).any()
    for seconds in [0,-.1,1.1,float('nan'),.501]:
        with pytest.raises(ValueError):action_start_step(480,240,seconds)


def test_zero_actions_match_v4_every_step_rewards_history_and_targets():
    a,b=fake(V4Env),fake(V5Env)
    for _ in range(4):
        oa,ra,da,ia=a.step(np.zeros((6,16)));ob,rb,db,ib=b.step(np.zeros((6,16)))
        np.testing.assert_array_equal(a.last_target,b.last_target)
        np.testing.assert_array_equal(oa,ob);np.testing.assert_array_equal(ra,rb)
        for k in ia:
            if k!='learning_mask':assert ia[k]==ib[k]
        assert da==db


def test_nonzero_prefix_ignored_preload_learns_without_extra_live_reward():
    e=fake(V5Env);action=np.full((6,16),.5)
    assert not e.learning_mask().any()
    e.step(action)
    np.testing.assert_allclose(e.last_target,e.current_reference,atol=1e-6)
    assert e.learning_mask().all()
    _,reward,_,info=e.step(action)
    assert info['learning_mask']==[True]*6
    assert np.max(abs(e.controller.offset))>0
    np.testing.assert_array_equal(reward,np.zeros(6))
    offset=e.controller.offset.copy();e.step(action)
    assert np.max(abs(e.controller.offset))>=np.max(abs(offset))


def test_failure_in_preload_is_terminal_and_absorbing():
    e=fake(V5Env);e.step(np.zeros((6,16)))
    original=e.contact
    def contact():
        original();e.depth[0]=.0011
    e.contact=contact
    _,r,_,info=e.step(np.ones((6,16)))
    assert r[0]==-5 and info['terminal'][0]
    assert not e.learning_mask()[0]
    saved=e.controller.offset[0].copy()
    _,r,_,_=e.step(-np.ones((6,16)))
    np.testing.assert_array_equal(saved,e.controller.offset[0]);assert r[0]==0


def test_v5_admission_launcher_routes_to_new_entrypoint(tmp_path):
    """CPU wiring fixture, not a newly simulated v5 baseline."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    source=root/'data/experiments/mug0_rl_baseline5_20261105_213034'
    plan=json.loads(subprocess.run([sys.executable,str(root/'scripts/run_isaacsim_residual_v5.py'),
        '--mode','plan','--object','mug','--candidate','0','--num-envs','6',
        '--output',str(tmp_path/'unused')],capture_output=True,text=True,check=True).stdout)
    fixture=tmp_path/'fixture';fixture.mkdir()
    for name in ['reference_snapshot.json','perturbations.json','baseline.json']:
        (fixture/name).write_bytes((source/name).read_bytes())
    (fixture/'config.json').write_text(json.dumps(plan['config']))
    routed=json.loads(subprocess.run([sys.executable,str(root/'scripts/train_isaacsim_reference.py'),
        '--baseline-results',str(fixture),'--num-envs','24','--iterations','200',
        '--plan-only','--output',str(tmp_path/'unused_train')],capture_output=True,text=True,check=True).stdout)
    assert routed['command'][1].endswith('isaacsim_residual_v5.sh')
    assert '--checkpoint' not in routed['command']
    assert routed['command'][routed['command'].index('--preload-seconds')+1]=='0.5'
    assert not (tmp_path/'unused_train').exists()
