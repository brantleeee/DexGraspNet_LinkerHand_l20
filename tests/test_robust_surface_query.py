import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robust_surface_query import closest_triangle_points,nearest_surface


def test_tiny_triangle_interior_projection_is_not_forced_to_edge():
    tri=np.array([[[0.,0,0],[.0005,0,0],[0,.0005,0]]])
    p=np.array([[.0001,.0001,.0002]])
    q=closest_triangle_points(tri,p)
    assert np.allclose(q,[[.0001,.0001,0]],atol=1e-14)


def test_surface_query_is_invariant_under_unit_change():
    v=np.array([[0.,0,0],[.0005,0,0],[0,.0005,0]])
    p=np.array([[.0001,.0001,.0002]])
    for scale in [.1,1,1000]:
        m=trimesh.Trimesh(v*scale,[[0,1,2]],process=False)
        q,d,face=nearest_surface(m,p*scale)
        assert np.allclose(q/scale,[[.0001,.0001,0]],atol=1e-13)
        assert np.allclose(d/scale,[.0002],atol=1e-13)
        assert face.tolist()==[0]
