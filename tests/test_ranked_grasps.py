import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_top_grasps import rank_candidates


def test_energy_order_stable_ties_and_no_pass_filter():
    data=[dict(energy=3),dict(energy=1,geometry_pass=False),dict(energy=1),dict(energy=2)]
    assert rank_candidates(data,3)==[1,2,3]


@pytest.mark.parametrize('k',[0,2])
def test_invalid_count_rejected(k):
    with pytest.raises(ValueError):rank_candidates([dict(energy=1)],k)


def test_nonfinite_energy_rejected():
    with pytest.raises(ValueError):rank_candidates([dict(energy=float('nan'))],1)
