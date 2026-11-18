import sys,itertools
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from l20_root_collision_policy import apply_root_policy,IGNORED


def test_only_eight_proximal_palm_pairs_removed_and_policy_idempotent():
    scene=ET.Element('mujoco');world=ET.SubElement(scene,'worldbody');names={'base_link','thumb_proximal'}|{f+'_'+part for f in ['index','middle','ring','pinky'] for part in ['metacarpals','proximal','middle','distal']}
    for n in names:ET.SubElement(ET.SubElement(world,'body',name=n),'geom',name=n)
    contact=ET.SubElement(scene,'contact')
    for a,b in itertools.combinations(names,2):ET.SubElement(contact,'pair',geom1=a,geom2=b)
    before=len(contact);assert len(apply_root_policy(scene,'angle-only'))==8
    remaining={frozenset((p.get('geom1'),p.get('geom2'))) for p in contact.findall('pair')}
    assert len(remaining)==before-8 and not remaining&IGNORED
    assert frozenset(('index_proximal','middle_proximal')) in remaining
    assert frozenset(('middle_distal','base_link')) in remaining
    assert frozenset(('thumb_proximal','base_link')) in remaining
    apply_root_policy(scene,'angle-only');assert len(contact.findall('exclude'))==8


def test_thumb_root_policy_keeps_distal_palm_and_cross_finger_pairs():
    scene=ET.Element('mujoco');world=ET.SubElement(scene,'worldbody')
    names={'base_link'}|{f+'_'+part for f in ['index','middle','ring','pinky'] for part in ['metacarpals','proximal','distal']}|{'thumb_metacarpals','thumb_metacarpals_base1','thumb_metacarpals_base2','thumb_proximal','thumb_distal'}
    for n in names:ET.SubElement(ET.SubElement(world,'body',name=n),'geom',name=n)
    contact=ET.SubElement(scene,'contact')
    for a,b in itertools.combinations(names,2):ET.SubElement(contact,'pair',geom1=a,geom2=b)
    assert len(apply_root_policy(scene,'all-roots-angle-only'))==12
    pairs={frozenset((x.get('geom1'),x.get('geom2'))) for x in contact.findall('pair')}
    assert frozenset(('thumb_metacarpals','base_link')) not in pairs
    assert frozenset(('thumb_proximal','thumb_metacarpals')) not in pairs
    assert frozenset(('thumb_distal','base_link')) in pairs
    assert frozenset(('thumb_proximal','index_proximal')) in pairs
