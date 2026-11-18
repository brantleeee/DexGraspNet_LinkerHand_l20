"""Object staging must preserve the user frame and must not normalize twice."""
from pathlib import Path
import sys
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from graspqp_drill_pipeline import stage_object_mesh


def test_identity_preserves_file_and_offset(tmp_path):
    mesh=trimesh.creation.box(extents=[.2555,.2234,.075]);mesh.apply_translation([.3,-.2,.1])
    src=tmp_path/'source.obj';mesh.export(src)
    obj,info=stage_object_mesh(src,1.,tmp_path/'input')
    assert src.read_bytes()==(tmp_path/'input/remeshed.obj').read_bytes()
    assert np.allclose(obj.bounds,mesh.bounds)
    assert info['source_scale']==1. and info['physics_admitted']


def test_open_mesh_not_admitted_for_physics(tmp_path):
    mesh=trimesh.creation.box();mesh.update_faces(np.arange(len(mesh.faces)-1))
    src=tmp_path/'open.obj';mesh.export(src)
    _,info=stage_object_mesh(src,.1,tmp_path/'input')
    assert not info['object_watertight'] and not info['physics_admitted']
    assert info['sdf_sign']=='winding_number'
    assert np.allclose(info['object_extents_m'],[.1,.1,.1])
