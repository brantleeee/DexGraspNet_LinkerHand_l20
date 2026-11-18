import copy
import numpy as np
import pytest
from isaacsim_grasp.paired_evaluation import make_samples,validate_samples,FixedUniform,compare_initial_states,summarize_pairs,CASES


def test_explicit_samples_repeat_all_six_directions_and_stay_in_bounds():
    a=make_samples(5,12,.0005,20261105);b=make_samples(5,12,.0005,20261105)
    x=validate_samples(a,5,12,.0005)
    np.testing.assert_array_equal(x,np.array(b['offsets_m']))
    assert x.shape==(5,12,3) and np.max(abs(x))<=.0005
    assert np.array_equal(x[:,0],x[:,5]) and np.array_equal(x[:,6],x[:,11])
    assert not np.array_equal(x[:,0],x[:,6])
    assert not np.array_equal(x,np.array(make_samples(5,12,.0005,0)['offsets_m']))


def test_fixed_uniform_copies_exact_samples_and_checks_requested_bounds():
    x=np.array(make_samples(1,6)['offsets_m'][0]);sampler=FixedUniform(x,.0005)
    first=sampler.uniform(-.0005,.0005,(6,3));first[:]=0
    np.testing.assert_array_equal(sampler.uniform(-.0005,.0005,(6,3)),x)
    with pytest.raises(ValueError):sampler.uniform(-.001,.001,(6,3))


def test_invalid_sample_bounds_or_unpaired_direction_rejected():
    s=make_samples(1,6)
    s['offsets_m'][0][1][0]+=.001
    with pytest.raises(ValueError):validate_samples(s,1,6,.0005)
    s=make_samples(1,6);s['offsets_m'][0][1]=[0,0,0]
    with pytest.raises(ValueError):validate_samples(s,1,6,.0005)


def test_actual_initial_state_mismatch_is_not_silently_tolerated():
    s={'object':np.zeros((6,3)),'joint':np.ones((6,21))}
    assert all(v==0 for v in compare_initial_states(s,copy.deepcopy(s)).values())
    other=copy.deepcopy(s);other['joint'][0,0]+=1e-8
    with pytest.raises(ValueError):compare_initial_states(s,other)


def outcomes():
    return [dict(episode=e,env=i,group=e,case=case,offset_m=[e*.0001,0,0],passed=True) for e in range(2) for i,case in enumerate(CASES)]


def test_paired_statistics_use_trial_and_six_direction_group_denominators():
    zero=outcomes();policy=outcomes();zero[0]['passed']=False;policy[7]['passed']=False
    s=summarize_pairs(zero,policy)
    assert s['trials']==12 and s['six_direction_groups']==2
    assert s['outcomes']=={'both_pass':10,'policy_only':1,'zero_only':1,'both_fail':0}
    assert sum(g['zero_all_six'] for g in s['groups'])==1 and sum(g['policy_all_six'] for g in s['groups'])==1
    assert s['per_direction']['gx+']['zero_passed']==1


def test_different_samples_missing_trials_and_duplicate_trials_rejected():
    z=outcomes();p=outcomes();p[0]['offset_m']=[.0002,0,0]
    with pytest.raises(ValueError):summarize_pairs(z,p)
    with pytest.raises(ValueError):summarize_pairs(z,outcomes()[:-1])
    with pytest.raises(ValueError):summarize_pairs(z,outcomes()+[outcomes()[0]])
