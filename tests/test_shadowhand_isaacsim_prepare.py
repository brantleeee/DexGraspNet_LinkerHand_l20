"""CPU contract checks; fixtures are old mug poses, not drill stability evidence."""
import copy
import xml.etree.ElementTree as ET
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_shadowhand_isaacsim import prepare
SOURCE=ROOT/'external/DexGraspNet_official_compat/data/experiments/shadow_cup_jenga_006_20261102/results/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03.npy'
MESH=ROOT/'data/meshdata/power-drill/coacd/decomposed.obj'


def test_generation_fk_geometry_and_exact_saved_scale(tmp_path):
    data=copy.deepcopy(np.load(SOURCE,allow_pickle=True))
    xml=ET.parse(ROOT/'external/DexGraspNet_official_compat/grasp_generation/mjcf/shadow_hand_wrist_free.xml')
    for joint in xml.findall('.//worldbody//joint'):
        lo,hi=map(float,joint.get('range').split())
        for g in data:g['qpos'][joint.get('name')]=float(np.clip(g['qpos'][joint.get('name')],lo,hi))
    fixture=tmp_path/'fixture.npy';np.save(fixture,data)
    m=prepare(fixture,MESH,tmp_path/'inputs',top_k=16)
    assert len(m['joints'])==22 and len(m['links'])==23
    assert all(j['mimic'] is None for j in m['joints'])
    assert m['fk_max_matrix_error']<2e-6
    assert len(m['meshes'])==len([l for l in m['links'] if l['colliders']])+1
    assert m['scale']==float(np.load(SOURCE,allow_pickle=True)[0]['scale'])
    assert len(m['candidates'])==16
    assert [c['energy'] for c in m['candidates']]==sorted(c['energy'] for c in m['candidates'])
    with pytest.raises(ValueError,match='Output must be new'):
        prepare(SOURCE,MESH,tmp_path/'inputs',top_k=1)


def test_mixed_scale_is_rejected(tmp_path):
    data=np.load(SOURCE,allow_pickle=True)[:2]
    data[1]['scale']=.2555
    source=tmp_path/'mixed.npy';np.save(source,data)
    with pytest.raises(ValueError,match='one exact'):
        prepare(source,MESH,tmp_path/'inputs',indices=[0,1])
