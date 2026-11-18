import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco,trimesh,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from validate_grasp_dynamics import build_template


def test_native_sdf_template_has_no_python_hull_and_no_duplicate_mimics(tmp_path,monkeypatch):
    if not (ROOT/'data/meshdata/power-drill/coacd/decomposed.obj').exists():pytest.skip('Prepare power-drill asset first')
    def forbidden(_):raise AssertionError('SDF path must not replace drill with a hull')
    monkeypatch.setattr(trimesh.Trimesh,'convex_hull',property(forbidden))
    root,robot,count=build_template(ROOT,tmp_path,'power-drill',mesh_sdf=True)
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    active=(m.geom_contype!=0)|(m.geom_conaffinity!=0)
    assert count==1 and active.sum()==23 and np.all(m.geom_type[active]==mujoco.mjtGeom.mjGEOM_SDF)
    assert m.neq==5 and m.nu==16
