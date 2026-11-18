from pathlib import Path
import json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import check_shadowhand_grip_strength as mod


def baseline(tmp_path,mass):
    p=tmp_path/'base/inputs';p.mkdir(parents=True)
    m=dict(object_mass_kg=mass,candidates=[dict(index=30,targets=dict(j=.12),qpos=dict(WRJTx=.03))],joints=[dict(name='j',limit=dict(effort=1.))])
    (p/'manifest.json').write_text(json.dumps(m));(p/'meshes.npz').write_bytes(b'frozen mesh fixture')
    return p.parent,m


def test_strength_experiment_preserves_reference_and_frozen_source(tmp_path,monkeypatch):
    p,m=baseline(tmp_path,.6);old=(p/'inputs/manifest.json').read_bytes();out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['test','--baseline',str(p),'--output',str(out),'--plan-only']);mod.main()
    new=json.loads((out/'inputs/manifest.json').read_text())
    assert new['candidates']==m['candidates'] and new['joints']==m['joints']
    assert (out/'inputs/meshes.npz').read_bytes()==(p/'inputs/meshes.npz').read_bytes()
    assert (p/'inputs/manifest.json').read_bytes()==old


def test_mass_is_not_silently_changed(tmp_path,monkeypatch):
    p,_=baseline(tmp_path,.3);out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['test','--baseline',str(p),'--output',str(out),'--plan-only'])
    with pytest.raises(SystemExit):mod.main()
    assert not out.exists()
