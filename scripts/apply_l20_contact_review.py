"""Apply explicit user axes from frozen v5; leave unspecified middle points pending."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from contact_frame_display import contact_basis
ROOT=Path(__file__).resolve().parents[1]
REMOVED={139,150,223}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reframe(prior,rule):
    """Rule always uses the frozen input frame, never the new output axes."""
    f=copy.deepcopy(prior);axis=rule.lstrip('-')
    if axis not in ['Xc','Yc','Zc']:raise ValueError('Unknown axis')
    n=np.asarray(prior[axis])*(-1 if rule.startswith('-') else 1)
    n,t1,t2=contact_basis(n,[0,0,1]);quality=copy.deepcopy(prior['patch_quality'])
    quality.update(method='user selected axis from frozen v5',outward_probe_pass=None,
        previous_outward_probe_pass=prior['patch_quality'].get('outward_probe_pass'),
        support_role='historical_reference_only',dominant_area_fraction=None)
    if 'support_face_normal' in quality:
        quality['normal_to_reference_face_deg']=float(np.degrees(np.arccos(np.clip(np.dot(n,quality['support_face_normal']),-1,1))))
    f.update(Xc=t1.tolist(),Yc=t2.tolist(),Zc=n.tolist(),previous_patch_quality=prior['patch_quality'],
        previous_axes={k:prior[k] for k in ['Xc','Yc','Zc']},user_rule=rule,
        normal_source='user_confirmed_frozen_v5_axis',patch_quality=quality,needs_review=True,
        review_status='user_direction_set_surface_relation_pending')
    return f


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,default=ROOT/'data/hand_contacts/l20_contact_v5_20261117')
    p.add_argument('--middle-red-ids',nargs='*',type=int,default=[])
    p.add_argument('--review-json',type=Path,help='Frozen manifest hash and middle_rules {stable_id: Xc/-Xc/...}')
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new version directory')
    m=json.loads((a.previous/'manifest.json').read_text())
    if m['version']!='l20-facet-attached-v5-20261117':p.error('Rules explicitly reference frozen v5')
    for path,h in m['asset_sha256'].items():
        if sha(path)!=h:raise ValueError('Original source changed: '+path)
    for path,h in m['generated_sha256'].items():
        if sha(a.previous/path)!=h:raise ValueError('Frozen library changed: '+path)
    old=json.loads((a.previous/'contact_frames.json').read_text());by_id={x['stable_id']:x for x in old}
    middle={i:'Xc' for i in a.middle_red_ids}
    if a.review_json:
        review=json.loads(a.review_json.read_text())
        if review['source_manifest_sha256']!=sha(a.previous/'manifest.json'):p.error('Review references another source')
        middle.update({int(i):rule for i,rule in review['middle_rules'].items()})
    for i in middle:
        if i not in by_id or i in REMOVED or by_id[i]['parent_link']!='middle_distal':p.error('Invalid middle stable ID: '+str(i))
    rules={133:'-Yc',**{x['stable_id']:'-Zc' for x in old if x['parent_link']=='thumb_metacarpals'},**middle}
    removed=set(m['removed_stable_ids'])|REMOVED;frames=[]
    points={k:[] for k in json.loads((a.previous/'contact_points.json').read_text())}
    for prior in old:
        i=prior['stable_id']
        if i in removed:continue
        f=reframe(prior,rules[i]) if i in rules else copy.deepcopy(prior)
        if f['parent_link']=='middle_distal' and i not in middle:
            f.update(needs_review=True,review_status='middle_red_or_blue_unspecified')
        f['runtime_id']=len(frames);f['local_id']=len(points[f['parent_link']]);points[f['parent_link']].append(f['origin_P_m']);frames.append(f)
    new={x['stable_id']:x for x in frames};mapping=[]
    for entry in json.loads((a.previous/'id_mapping.json').read_text()):
        entry=copy.deepcopy(entry);i=entry['stable_id'];entry.update(removed=i in removed,runtime_id=new[i]['runtime_id'] if i in new else None)
        if i in new:entry['new_local_id']=new[i]['local_id']
        else:entry.pop('new_local_id',None)
        mapping.append(entry)
    a.output.mkdir(parents=True)
    for name,data in [('contact_points.json',points),('contact_frames.json',frames),('id_mapping.json',mapping)]:
        (a.output/name).write_text(json.dumps(data,indent=2)+'\n')
    manifest=dict(version='l20-user-review-v6-20261117',previous=str(a.previous.resolve()),previous_manifest_sha256=sha(a.previous/'manifest.json'),
        asset_sha256=m['asset_sha256'],point_count=len(frames),removed_stable_ids=sorted(removed),new_removals=sorted(REMOVED),
        manual_rules={str(i):r for i,r in rules.items()},middle_red_rules={str(i):r for i,r in middle.items()},
        middle_unspecified_ids=[x['stable_id'] for x in frames if x['review_status']=='middle_red_or_blue_unspecified'],
        implementation_sha256=sha(Path(__file__)),generated_sha256={k:sha(a.output/k) for k in ['contact_points.json','contact_frames.json','id_mapping.json']},
        note='User directions relative to frozen v5; not mesh outward certification. Unspecified middle points remain unchanged and pending. Default generator unchanged.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Complete:',a.output.resolve(),len(frames),'points; middle pending:',manifest['middle_unspecified_ids'])


if __name__=='__main__':main()
