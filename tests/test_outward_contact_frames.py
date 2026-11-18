from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from revise_l20_contact_library import oriented_distance_normal


def test_outside_and_inside_probes_produce_same_outward_direction():
    q=np.array([1.,0,0])
    assert np.allclose(oriented_distance_normal([1.01,0,0],q,False),[1,0,0])
    assert np.allclose(oriented_distance_normal([.99,0,0],q,True),[1,0,0])


def test_zero_offset_does_not_invent_a_normal():
    import pytest
    with pytest.raises(ValueError):oriented_distance_normal([0,0,0],[0,0,0],False)
