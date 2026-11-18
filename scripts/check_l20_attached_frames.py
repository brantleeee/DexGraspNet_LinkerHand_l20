"""Validate v5 geometry bindings, unchanged references and Canvas exports."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True,type=Path);p.add_argument('--viewer',required=True,type=Path);a=p.parse_args()
    m=json.loads((a.library/'manifest.json').read_text());f=json.loads((a.library/'contact_frames.json').read_text())
    previous={x['stable_id']:x for x in json.loads((Path(m['previous'])/'contact_frames.json').read_text())}
    for path,h in m['asset_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    for path,h in m['generated_sha256'].items():assert hashlib.sha256((a.library/path).read_bytes()).hexdigest()==h
    assert len(f)==406 and [x['runtime_id'] for x in f]==list(range(406))
    assert set(x['stable_id'] for x in f)==set(previous)-{271,273,280,284}
    points=json.loads((a.library/'contact_points.json').read_text());rebuilt=[]
    for x in f:
        prior=previous[x['stable_id']]
        assert np.array_equal(x['origin_P_m'],prior['origin_P_m'])
        assert np.array_equal(points[x['parent_link']][x['local_id']],x['origin_P_m'])
        basis=np.array([x[k] for k in ['Xc','Yc','Zc']]).T
        assert np.allclose(basis.T@basis,np.eye(3),atol=1e-9) and np.isclose(np.linalg.det(basis),1)
        if x['parent_link'] in m['rebuilt_links'] or x['stable_id']==337:
            rebuilt.append(x);q=x['patch_quality'];tri=np.array(q['support_triangle_m']);n=np.array(x['Zc'])
            edges=tri[[1,2]]-tri[0]
            assert np.allclose((edges/np.linalg.norm(edges,axis=1)[:,None])@n,0,atol=1e-8)
            assert q['outward_probe_pass'] is True
            assert x['needs_review']
        else:
            for k in ['Xc','Yc','Zc']:assert np.array_equal(x[k],prior[k])
    assert len(rebuilt)==57
    assert json.loads((a.viewer/'all_contact_frames.json').read_text())==f
    for link in ['middle_distal','ring_distal','thumb_metacarpals']:
        s=(a.viewer/(link+'.html')).read_text()
        data=json.loads(re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',s).group(1))
        for pt in data['points']:
            if link!='thumb_metacarpals' or pt['global_id']==337:assert len(pt['support_triangle_mm'])==3
    assert len(list(a.viewer.glob('*.svg')))==66
    result=dict(points=406,rebuilt_frames=57,explicit_removed_ids=[271,273,280,284],
        positions_unchanged=True,source_hashes_unchanged=True,frames_right_handed=True,
        support_face_perpendicular=True,centroid_outward_checks_passed=57,
        feature_edge_ambiguous=sum(x['patch_quality']['ambiguous_surface'] for x in rebuilt),
        viewer_and_library_identical=True,P337_link=next(x['parent_link'] for x in f if x['stable_id']==337),
        P337_intended_surface_confirmed=False,note='Facet alignment is not grasp-region or dynamics certification.')
    (a.viewer/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
