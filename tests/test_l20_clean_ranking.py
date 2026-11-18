from pathlib import Path
import sys,json,hashlib
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_l20_clean_drill import audit_ranking


def fixture(tmp_path):
    source=tmp_path/'poses.npy';np.save(source,np.array([dict(scale=1.,energy=3.),dict(scale=1.,energy=1.)],dtype=object))
    rank=dict(object_code='power-drill-clean-l20',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=[dict(rank=1,candidate_index=1,scale=1.,energy=1.,geometry_pass=False),dict(rank=2,candidate_index=0,scale=1.,energy=3.,geometry_pass=True)])
    p=tmp_path/'ranking.json';p.write_text(json.dumps(rank));return p,source,rank


def test_rank_identity_preserved_without_geometry_filter(tmp_path):
    p,_,_=fixture(tmp_path);_,_,_,rows,indices=audit_ranking(p,2)
    assert indices==[1,0] and not rows[0]['geometry_pass'] and len(rows)==2


def test_mismatched_rank_energy_is_rejected(tmp_path):
    p,_,rank=fixture(tmp_path);rank['rows'][0]['energy']=.1;p.write_text(json.dumps(rank))
    with pytest.raises(ValueError,match='energy or saved scale'):audit_ranking(p,2)
