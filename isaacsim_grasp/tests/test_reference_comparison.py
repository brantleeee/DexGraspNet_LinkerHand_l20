import pytest
from isaacsim_grasp.reference_comparison import summarize


def test_reference_comparison_counts_samples_not_repeats():
    rows=[]
    for group in range(2):
        for case in ['gx+','gx-','gy+','gy-','gz+','gz-']:
            rows.append(dict(group=group,case=case,original_passed=not(group==0 and case=='gy+'),
                refined_passed=not(group==1 and case=='gz+')))
    s=summarize(rows)
    assert s['translation_samples']==2 and s['trials_per_reference']==12
    assert s['original_passed']==s['refined_passed']==11
    assert s['original_all_six']==s['refined_all_six']==1
    assert s['outcomes']==dict(both_pass=10,refined_only=1,original_only=1,both_fail=0)
    with pytest.raises(ValueError):summarize(rows[:-1])
