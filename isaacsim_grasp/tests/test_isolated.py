import numpy as np
import pytest
from isaacsim_grasp.isolated import select_group,compare_traces,contact_events
from isaacsim_grasp.paired_evaluation import make_samples


def test_select_one_group_preserves_six_directions_and_offsets():
    s=make_samples(20,6,seed=42);one=select_group(s,18)
    assert one['episodes']==1 and one['source_group']==18
    np.testing.assert_array_equal(one['offsets_m'][0],s['offsets_m'][18])
    with pytest.raises(ValueError):select_group(s,20)


def trace():
    return dict(time=np.array([0.,1.,2.5,3.]),object_position=np.zeros((4,1,3)),
        object_quaternion=np.tile([1.,0,0,0],(4,1,1)),joint_position=np.zeros((4,1,2)),
        joint_target=np.zeros((4,1,2)),contact_force=np.zeros((4,1)),object_linear_velocity=np.zeros((4,1,3)))


def test_same_initial_state_is_not_enough_and_prefix_ignores_later_actions():
    a,b=trace(),trace();assert compare_traces(a,b)['passed']
    b['object_position'][3,0,0]=.1;assert compare_traces(a,b)['passed']
    assert not compare_traces(a,b,13.)['passed']
    b['object_position'][2,0,0]=.001;assert not compare_traces(a,b)['passed']
    assert compare_traces(a,b)['max_target_difference_rad']==0


def test_contact_loss_has_minimum_gap_not_single_frame_flicker():
    t=np.arange(40)/240;active=np.zeros(40,bool);active[3:10]=True;active[11:15]=True
    x=contact_events(t,active)
    assert x['first_contact_seconds']==3/240
    gaps=[e for e in x['events'] if e['event']=='contact_gap']
    assert len(gaps)==1 and gaps[0]['time_seconds']==15/240


def test_detailed_contacts_keep_filter_and_point_coordinates(tmp_path):
    from types import SimpleNamespace
    from isaacsim_grasp.tests.test_diagnostics import fake_env
    from isaacsim_grasp.finger_diagnostics import FingerDiagnostics
    e=fake_env();e.ref['hand_links']=['thumb_tip','index_tip','base_link'];e.cfg={'residual_scale':.15}
    e.rot=np.tile(np.eye(3),(2,1,1));e.origins=np.array([[10.,0,0],[20.,0,0]])
    counts=np.array([[1,0,0],[0,1,0]]);starts=np.array([[0,0,0],[0,1,0]])
    e.obj=SimpleNamespace(get_contact_force_data=lambda dt:(np.array([[2.],[3.]]),np.array([[10.1,0,0],[20.2,0,0]]),
        np.array([[1.,0,0],[0,1.,0]]),np.array([[-.0001],[-.0002]]),counts,starts),
        get_contact_force_matrix=lambda dt:np.zeros((2,3,3)),
        get_friction_data=lambda dt:(np.array([[.1,0,0]]),np.array([[10.1,0,0]]),np.array([[1,0,0],[0,0,0]]),np.zeros((2,3),int)))
    d=FingerDiagnostics(e);d.begin()
    for k in range(1,3):e.physics_step=k;e.contact()
    result=d.finish(tmp_path/'trajectory.npz')
    with np.load(tmp_path/'contact_points.npz') as x:
        np.testing.assert_allclose(x['normal_contacts'][:2,3],[.1,.2])
        np.testing.assert_array_equal(x['normal_contacts'][:2,1:3],[[0,0],[1,1]])
    with np.load(tmp_path/'trajectory.npz') as x:
        assert x['finger_normal_force'][1,0,0]==2
        assert x['finger_normal_force'][1,1,1]==3
    assert len(result['finger_contact_events'])==2
