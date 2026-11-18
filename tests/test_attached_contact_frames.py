from pathlib import Path
import sys,numpy as np,trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rebuild_l20_attached_frames import supporting_face


def test_corner_uses_a_real_plane_and_reports_ambiguity():
    mesh=trimesh.Trimesh([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,1,2],[0,3,1]],process=False)
    q,n,e=supporting_face(mesh,[.2,-.1,-.1])
    assert np.allclose(q,[.2,0,0])
    tri=np.array(e['support_triangle_m'])
    assert abs(np.dot(n,tri[1]-tri[0]))<1e-10
    assert abs(np.dot(n,tri[2]-tri[0]))<1e-10
    assert e['ambiguous_surface'] and e['on_feature_edge']


def test_nearby_opposite_sheet_is_not_averaged_into_normal():
    mesh=trimesh.Trimesh([[0,0,0],[1,0,0],[0,1,0],[0,0,.001],[0,1,.001],[1,0,.001]],[[0,1,2],[3,4,5]],process=False)
    q,n,e=supporting_face(mesh,[.2,.2,.0001])
    assert np.allclose(q,[.2,.2,0]) and np.allclose(n,[0,0,1])
    assert not e['ambiguous_surface']
