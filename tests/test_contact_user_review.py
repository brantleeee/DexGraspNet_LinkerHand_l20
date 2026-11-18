import copy,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from apply_l20_contact_review import reframe


def test_user_rotation_invalidates_old_geometry_certificate_without_changing_source():
    old=dict(Xc=[1,0,0],Yc=[0,1,0],Zc=[0,0,1],
        patch_quality=dict(outward_probe_pass=True,support_face_normal=[0,0,1]))
    frozen=copy.deepcopy(old);new=reframe(old,'-Yc')
    assert old==frozen and np.allclose(new['Zc'],[0,-1,0])
    assert new['patch_quality']['outward_probe_pass'] is None
    assert new['patch_quality']['previous_outward_probe_pass'] is True
    assert new['patch_quality']['normal_to_reference_face_deg']==90


def test_normal_flip_rebuilds_a_right_handed_frame():
    old=dict(Xc=[1,0,0],Yc=[0,1,0],Zc=[0,0,1],patch_quality={})
    new=reframe(old,'-Zc');basis=np.array([new[k] for k in ['Xc','Yc','Zc']]).T
    assert np.allclose(new['Zc'],[0,0,-1])
    assert np.allclose(basis.T@basis,np.eye(3)) and np.isclose(np.linalg.det(basis),1)
