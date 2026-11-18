"""Explicit four-finger proximal/palm assembly collision policy."""
import xml.etree.ElementTree as ET
FINGERS=('index','middle','ring','pinky')
IGNORED={frozenset((f+'_proximal',p)) for f in FINGERS for p in ('base_link',f+'_metacarpals')}
THUMB_ROOTS={frozenset(('thumb_metacarpals',p)) for p in ('base_link','thumb_metacarpals_base1','thumb_metacarpals_base2','thumb_proximal')}


def apply_root_policy(scene,policy):
    if policy=='all':return []
    if policy not in ('angle-only','all-roots-angle-only'):raise ValueError('Unknown root collision policy')
    ignored=IGNORED|THUMB_ROOTS if policy=='all-roots-angle-only' else IGNORED
    owner={g.get('name'):b.get('name') for b in scene.findall('.//body') for g in b.findall('geom')}
    contact=scene.find('contact')
    if contact is None:contact=ET.SubElement(scene,'contact')
    for pair in list(contact.findall('pair')):
        if frozenset((owner.get(pair.get('geom1')),owner.get(pair.get('geom2')))) in ignored:contact.remove(pair)
    existing={frozenset((x.get('body1'),x.get('body2'))) for x in contact.findall('exclude')}
    for pair in sorted(ignored,key=lambda x:sorted(x)):
        if pair not in existing:
            left,right=sorted(pair);ET.SubElement(contact,'exclude',body1=left,body2=right)
    return [sorted(p) for p in sorted(ignored,key=lambda x:sorted(x))]
