import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_l20_middle_self_collision import triangle_overlap


def test_coplanar_and_separated_triangles():
    a=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])
    assert triangle_overlap(a,a).item()
    assert not triangle_overlap(a,a+[2,2,0]).item()
    assert not triangle_overlap(a,a+[0,0,1]).item()


def test_crossing_and_touching():
    a=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])
    b=np.array([[[.2,.2,-1],[.2,.2,1],[.8,.2,0]]])
    assert triangle_overlap(a,b).item()
    assert triangle_overlap(a,a+[1,0,0]).item()
