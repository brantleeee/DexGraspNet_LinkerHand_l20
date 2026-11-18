from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import trimesh
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dexgraspnet_drill_regions import read_regions


def fixture(tmp_path):
    mesh=trimesh.creation.box();source=tmp_path/'full.obj';mesh.export(source)
    mesh=trimesh.load(source,force='mesh',process=False)
    digest=hashlib.sha256(source.read_bytes()).hexdigest();labels=np.arange(len(mesh.faces))%5
    np.savez(tmp_path/'semantic_surface.npz',vertices=mesh.vertices,faces=mesh.faces,face_labels=labels,obj_sha256=digest)
    classes={}
    for i in range(5):
        name=f'ids_{i}.npy';np.save(tmp_path/name,np.flatnonzero(labels==i));classes[str(i)]=dict(class_id=i,face_ids=name)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(source_obj=str(source),source_sha256=digest,classes=classes)))
    return source


def test_labels_require_same_geometry_and_original_face_ids(tmp_path):
    fixture(tmp_path);_,mesh,v,f,labels=read_regions(tmp_path)
    assert len(labels)==len(mesh.faces) and (labels==0).any()
    np.save(tmp_path/'ids_0.npy',np.array([1]))
    with pytest.raises(ValueError,match='face map'):read_regions(tmp_path)


def test_changed_source_is_rejected(tmp_path):
    source=fixture(tmp_path);source.write_text(source.read_text()+'\n# changed\n')
    with pytest.raises(ValueError,match='source mesh changed'):read_regions(tmp_path)
