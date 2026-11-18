from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_middle_pair_gaps import normalized_triangle_distance


def test_parallel_microtriangles_keep_gap_under_scale_and_translation():
    a=np.array([[[0.,0,0],[1,0,0],[0,1,0]]]);b=a+np.array([0,0,.25])
    for scale in [1.,1e-3,1e-5]:
        for offset in [np.zeros(3),np.array([.1,-.2,.3])]:
            distance=normalized_triangle_distance(a*scale+offset,b*scale+offset)
            assert np.allclose(distance,[.25*scale],atol=1e-12,rtol=1e-6)


def test_intersecting_microtriangles_stay_zero_after_normalization():
    a=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])*1e-4
    b=np.array([[[.2,.2,-1],[.2,.2,1],[.5,.2,0]]])*1e-4
    assert normalized_triangle_distance(a,b)[0]==0
