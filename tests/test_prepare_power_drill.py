"""Check bore repair preserves the tunnel rather than capping its openings."""
import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_power_drill import close_matching_bore


def test_bridge_bore_adds_inner_wall_not_end_caps():
    solid=trimesh.creation.annulus(r_min=.02,r_max=.03,height=.016,sections=40)
    # Remove inner cylindrical wall while retaining outer wall and both annuli.
    r=np.linalg.norm(solid.vertices[:,:2],axis=1)
    inner=np.all(np.isclose(r[solid.faces],.02),axis=1)
    opened=trimesh.Trimesh(solid.vertices,solid.faces[~inner],process=False)
    repaired,added=close_matching_bore(opened)
    assert added==80 and repaired.is_volume
    assert np.allclose(repaired.bounds,solid.bounds)
    assert np.isclose(repaired.volume,solid.volume,rtol=1e-6)
