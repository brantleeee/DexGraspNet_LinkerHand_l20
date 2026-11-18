import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from visualize_official_shadowhand_normals import capsule_surface


def test_capsule_side_and_cap_outward_normals():
    q,n,d=capsule_surface(np.array([[0,-.007,.012],[0,0,.032]]),.007,.024)
    assert np.allclose(n,[[0,-1,0],[0,0,1]])
    assert np.allclose(q,[[0,-.007,.012],[0,0,.031]])
    assert np.allclose(d,[0,.001])


def test_positive_inside_sdf_gradient_is_negative_outward_normal():
    p=np.array([[.003,-.008,.01]]);_,n,_=capsule_surface(p,.007,.024)
    derivative=[]
    for axis in np.eye(3):
        _,_,hi=capsule_surface(p+axis*1e-7,.007,.024)
        _,_,lo=capsule_surface(p-axis*1e-7,.007,.024)
        derivative.append(float(-(hi-lo)[0]/2e-7))
    assert np.allclose(derivative,-n[0],atol=1e-8)
