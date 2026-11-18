from pathlib import Path
import sys
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_clean_graspqp_drill import closed_proxy


def test_nonconvex_union_keeps_frame_bounds_and_source():
    # L-shaped union must retain its concavity, rather than become its convex hull.
    a=trimesh.creation.box(extents=[.04,.02,.02]);a.apply_translation([.13,-.2,.3])
    b=trimesh.creation.box(extents=[.02,.04,.02]);b.apply_translation([.12,-.19,.3])
    source=trimesh.util.concatenate([a,b]);vertices=source.vertices.copy();faces=source.faces.copy()
    solid,audit=closed_proxy(source)
    assert solid.is_volume
    assert np.allclose(solid.bounds,source.bounds,atol=2e-7)
    assert solid.volume<solid.convex_hull.volume*.99
    assert not audit['convex_hull'] and not audit['convex_decomposition']
    assert np.array_equal(source.vertices,vertices) and np.array_equal(source.faces,faces)
