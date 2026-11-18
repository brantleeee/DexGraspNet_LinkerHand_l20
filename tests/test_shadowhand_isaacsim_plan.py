"""Check binding of explicit physical mesh, assumed mass, and frozen run plan."""
from pathlib import Path
import sys,json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import validate_shadowhand_isaacsim as runner
from prepare_shadowhand_isaacsim import sha


def test_explicit_mesh_metadata_and_prepared_plan(tmp_path,monkeypatch):
    source=tmp_path/'power-drill.npy';np.save(source,np.array([dict(scale=1.,energy=1.,qpos={},semantic_verified=False)],dtype=object))
    mesh=tmp_path/'physical.obj';mesh.write_text('explicit mesh fixture')
    meta=tmp_path/'meta.json';mass=.6;extent=np.array([.25,.22,.075]);diagonal=mass/12*np.array([sum(extent[[1,2]]**2),sum(extent[[0,2]]**2),sum(extent[[0,1]]**2)])
    meta.write_text(json.dumps(dict(mass_kg_assumed=mass,diaginertia_kg_m2=diagonal.tolist(),inertia_model='uniform bounding box, approximate')))
    out=tmp_path/'out'
    def prepare(src,actual,folder,indices,topk):
        assert actual==mesh.resolve() and src==source.resolve()
        folder.mkdir(parents=True)
        return dict(source_sha256=sha(src),object_extents_m=extent.tolist(),object_mass_kg=.58,candidates=[dict(index=0)],meshes=[dict(key='object',sha256=sha(actual))])
    monkeypatch.setattr(runner,'prepare',prepare)
    args=['test','--result-file',str(source),'--object-mesh',str(mesh),'--physical-metadata',str(meta),'--top-k','1','--prepare-only','--output',str(out)]
    monkeypatch.setattr(sys,'argv',args);runner.main()
    m=json.loads((out/'inputs/manifest.json').read_text());assert m['object_mass_kg']==.6 and not m['mass_is_measured']
    monkeypatch.setattr(sys,'argv',args+['--prepared-run']);runner.main()
    monkeypatch.setattr(sys,'argv',args+['--prepared-run','--gravity-seconds','11'])
    with pytest.raises(SystemExit):runner.main()
