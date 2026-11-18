import sys,json
from pathlib import Path
import numpy as np
import trimesh
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from admit_l20_middle_contact import decision,outside_query,binding_id,verify_binding,sha


def test_admission_requires_all_checks_and_never_uses_absolute_inside_distance():
    good=dict(label='palmar',reviewed=True)
    assert decision(good,.5,False,True)=='contact_candidate_admitted'
    assert decision(good,.1,True,True)=='inside_physics_proxy'
    assert decision(good,1.1,False,True)=='outside_contact_band'
    assert decision(good,.1,False,False)=='source_geometry_rejected'
    assert decision(dict(label='palmar',reviewed=False),.1,False,True)=='region_not_reviewed'
    assert decision(dict(label='unknown',reviewed=True),.1,False,True)=='region_not_allowed'


def test_reliable_proxy_query_distinguishes_inside_from_outside():
    box=trimesh.creation.box(extents=[.01]*3)
    distance,inside=outside_query(box,np.array([[.0055,0,0],[.0045,0,0]]))
    assert np.allclose(distance,[.5,.5])
    assert inside.tolist()==[False,True]


def test_annotation_binding_rejects_asset_changes(tmp_path):
    asset=tmp_path/'asset';asset.write_text('original')
    binding={'files':{str(asset):sha(asset)}}
    meta=dict(binding=binding,review_id=binding_id(binding));verify_binding(meta)
    asset.write_text('modified')
    with pytest.raises(ValueError,match='changed'):verify_binding(meta)
