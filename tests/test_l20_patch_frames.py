from pathlib import Path
import sys,numpy as np,trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rebuild_l20_contact_frames import patch_normal


def test_patch_anchor_distinguishes_opposing_nearby_surfaces():
    vertices=np.array([[0,0,0],[.001,0,0],[0,.001,0],[0,0,.0001],[0,.001,.0001],[.001,0,.0001]])
    mesh=trimesh.Trimesh(vertices,[[0,1,2],[3,4,5]],process=False)
    plus,q=patch_normal(mesh,np.array([.0002,.0002,0]),.002,np.array([0,0,1]))
    minus,_=patch_normal(mesh,np.array([.0002,.0002,0]),.002,np.array([0,0,-1]))
    assert plus[2]>.99 and minus[2]<-.99
    assert q['needs_review']  # opposed-sheet estimates must not silently become certified


def test_area_support_is_not_dominated_by_many_tiny_faces():
    v=[[0,0,0],[.002,0,0],[0,.002,0]];f=[[0,1,2]]
    for i in range(10):
        x=.0002+i*.00005;offset=len(v)
        v.extend([[x,0,.0001],[x,.00005,.0001],[x+.00005,0,.0001]])
        f.append([offset,offset+1,offset+2])
    mesh=trimesh.Trimesh(v,f,process=False)
    normal,quality=patch_normal(mesh,np.array([.0002,.0002,0]),.004)
    assert normal[2]>.99 and quality['dominant_area_fraction']>.9
