import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mesh_gap import triangle_distance,truncated_gap,contained


def test_triangle_distance_parallel_and_intersecting():
    a=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])
    assert np.allclose(triangle_distance(a,a+[0,0,.003]),.003)
    assert np.allclose(triangle_distance(a,a),0.)


def test_box_gap_and_far_clipping():
    a=trimesh.creation.box(extents=[.01]*3);b=a.copy();b.apply_translation([.011,0,0])
    assert abs(truncated_gap(a,b,.002)-.001)<1e-8
    b.apply_translation([.01,0,0]);assert truncated_gap(a,b,.002)==.002


def test_full_containment_and_open_target_unknown():
    outer=trimesh.creation.box(extents=[.01]*3);inner=trimesh.creation.box(extents=[.002]*3)
    assert contained(inner,outer)
    outer.update_faces(np.arange(len(outer.faces)-1))
    assert contained(inner,outer) is None
