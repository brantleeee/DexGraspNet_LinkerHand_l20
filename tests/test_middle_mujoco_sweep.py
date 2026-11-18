import sys
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_middle_mujoco_full_hand import path_angles,PATHS


def test_every_path_covers_its_endpoint_without_angular_jumps():
    ends={'mcp_only':[1.4,0],'pip_only':[0,1.57],'coupled':[1.4,1.57],'mcp_then_pip':[1.4,1.57],'pip_then_mcp':[1.4,1.57]}
    for kind in PATHS:
        q=path_angles(kind,.01)
        assert np.allclose(q[0],[0,0]) and np.allclose(q[-1],ends[kind])
        assert np.abs(np.diff(q,axis=0)).max()<=.01000001
        assert (q[:,1]*.8917<=1.4).all()


def test_explicit_sdf_pair_checks_welded_links_despite_disabled_masks():
    import trimesh
    box=trimesh.creation.box(extents=[.02,.02,.02])
    vertices=' '.join(map(str,box.vertices.ravel()));faces=' '.join(map(str,box.faces.ravel()))
    xml=f'''<mujoco><asset><mesh name="m" vertex="{vertices}" face="{faces}"/></asset>
    <worldbody><body name="a"><geom name="g1" type="sdf" mesh="m" contype="0" conaffinity="0"/>
    <body name="b" pos=".015 0 0"><geom name="g2" type="sdf" mesh="m" contype="0" conaffinity="0"/></body></body></worldbody>
    <contact><pair geom1="g1" geom2="g2"/></contact></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    assert model.npair==1 and model.body_weldid[1]==model.body_weldid[2]
    assert data.ncon>0 and min(c.dist for c in data.contact)<0
