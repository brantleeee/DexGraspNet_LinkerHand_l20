"""Check v4 stable IDs, frozen user rules, surface projection and viewer exports."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True,type=Path)
    p.add_argument('--viewer',required=True,type=Path);a=p.parse_args()
    m=json.loads((a.library/'manifest.json').read_text())
    f=json.loads((a.library/'contact_frames.json').read_text())
    old={x['stable_id']:x for x in json.loads((Path(m['previous'])/'contact_frames.json').read_text())}
    for path,h in m['asset_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    for path,h in m['generated_sha256'].items():assert hashlib.sha256((a.library/path).read_bytes()).hexdigest()==h,path
    assert len(f)==m['point_count']==410
    assert set(x['stable_id'] for x in f)==set(range(420))-set(m['removed_stable_ids'])
    assert [x['runtime_id'] for x in f]==list(range(410))
    points=json.loads((a.library/'contact_points.json').read_text())
    for x in f:
        basis=np.array([x[k] for k in ['Xc','Yc','Zc']]).T
        assert np.allclose(basis.T@basis,np.eye(3),atol=1e-8)
        assert np.isclose(np.linalg.det(basis),1)
        assert np.allclose(points[x['parent_link']][x['local_id']],x['origin_P_m'])
        if x['parent_link'] in m['projected_links']:
            assert np.linalg.norm(np.array(x['origin_P_m'])-x['surface_projection_Q_m'])<1e-10
        else:assert np.array_equal(x['origin_P_m'],x['original_P_m'])
        i=x['stable_id']
        target=-np.array(old[i]['Zc']) if i in [14,20] else (old[i]['Yc'] if i==297 else None)
        if target is not None:assert np.allclose(x['Zc'],target)
        elif 'user_confirmed' in old[i]['normal_source']:assert np.allclose(x['Zc'],old[i]['Zc'])
        if x['patch_quality']['outward_probe_pass'] is False:assert x['needs_review']
    shown=json.loads((a.viewer/'all_contact_frames.json').read_text())
    assert shown==f
    rows=json.loads((a.viewer/'all_contact_coordinates.json').read_text())
    assert len(rows)==410
    for r,x in zip(rows,f):assert np.allclose([r[k+'_m'] for k in 'xyz'],x['origin_P_m'])
    assert len(list(a.viewer.glob('*.svg')))==66
    assert 'getContext(\'2d\')' in (a.viewer/'viewer.js').read_text() or 'getContext("2d")' in (a.viewer/'viewer.js').read_text()
    summary=dict(points=len(f),removed=m['removed_stable_ids'],orthonormal_frames=True,
        source_hashes_unchanged=True,user_rules_preserved=True,viewer_matches_library=True,
        outward_status=dict(Counter(str(x['patch_quality']['outward_probe_pass']) for x in f)),
        outward_conflict_ids=[x['stable_id'] for x in f if x['patch_quality']['outward_probe_pass'] is False],
        projected_point_max_move_mm=max(x['position_change_mm'] for x in f),
        note='Data integrity and local orientation checks, not anatomical region or grasp stability certification.')
    (a.viewer/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
