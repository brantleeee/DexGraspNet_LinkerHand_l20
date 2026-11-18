import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from contact_frame_display import contact_basis,override_basis


def test_basis_is_orthonormal_and_right_handed():
    n,t1,t2=contact_basis([2,3,4],[1,-1,2])
    basis=np.column_stack([t1,t2,n])
    assert np.allclose(basis.T@basis,np.eye(3),atol=1e-12)
    assert np.isclose(np.linalg.det(basis),1.)


def test_parallel_edge_uses_valid_tangent_fallback():
    n,t1,t2=contact_basis([0,0,1],[0,0,2])
    assert np.isclose(n@t1,0.)
    assert np.allclose(np.cross(t1,t2),n)


def test_user_override_uses_frozen_direction_and_keeps_right_handed_basis():
    oldn,oldt1,oldt2=contact_basis([.2,.3,.4],[1,0,0])
    for desired in [-oldt1,oldt2]:
        record=dict(N=desired.tolist(),reference_N=oldn.tolist())
        n,t1,t2=override_basis(record)
        assert np.allclose(n,desired)
        assert np.allclose(np.cross(t1,t2),n)
        assert np.allclose(np.column_stack([n,t1,t2]).T@np.column_stack([n,t1,t2]),np.eye(3))
        assert np.allclose(override_basis(record)[0],desired)
