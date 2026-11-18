import numpy as np
from isaacsim_grasp.contact_timing import summarize_contact_timing


def trace():
    t=np.arange(240,321)/100;n=len(t)
    return dict(time=t,dt=.01,gravity_start_seconds=3.,finger_normal_force=np.zeros((n,1,2)),
        finger_contact_count=np.ones((n,1,2),int),link_normal_force_magnitude_sum=np.zeros((n,1,2)),
        link_contact_count=np.ones((n,1,2),int),object_position=np.zeros((n,1,3)),
        joint_position=np.zeros((n,1,1)),joint_target=np.zeros((n,1,1)),independent_ids=[0])


def load(z,mask):
    z['finger_normal_force'][mask,0,0]=.1
    z['link_normal_force_magnitude_sum'][mask,0,0]=.1


def analyze(z):return summarize_contact_timing(z,['index_distal','base_link'],['index','palm'],0)


def test_zero_force_records_do_not_count_as_load():
    r=analyze(trace())
    assert r['classification']=='no_sustained_finger_load_before_gravity'
    assert r['snapshots'][0]['links']['index_distal']['records']==1
    assert r['preload_longest_loaded_duration_seconds']['index']==0


def test_preload_loss_is_distinguished_from_postgravity_loss():
    z=trace();load(z,(z['time']>=2.6)&(z['time']<2.8))
    r=analyze(z)
    assert r['classification']=='preload_finger_load_lost_before_gravity'
    assert any(e['event']=='lost' and e['signal']=='load' and e['time_seconds']<3 for e in r['contact_transitions'])
    z=trace();load(z,(z['time']>=2.6)&(z['time']<3.02))
    assert analyze(z)['classification']=='finger_load_lost_after_gravity'


def test_transient_force_is_not_sustained():
    z=trace();load(z,z['time']==2.7)
    assert analyze(z)['classification']=='no_sustained_finger_load_before_gravity'
