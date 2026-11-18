import copy,json
from pathlib import Path
import numpy as np
import pytest
from isaacsim_grasp.reference_variants import variant_targets,proposals
from isaacsim_grasp.control import ResidualTargets


def reference():
    p=Path('data/experiments/mug0_v5_half_confirm20_20261106_135255/reference_snapshot.json')
    return json.loads(p.read_text())


def test_independent_delta_updates_mimic_without_changing_source():
    r=reference();old=copy.deepcopy(r)
    variants=proposals(r,['index_pip'],.01);v=next(v for v in variants if v['joint_deltas_rad']['index_pip']>0)
    targets=variant_targets(r,v)
    assert targets['index_pip']==pytest.approx(r['targets']['index_pip']+.01)
    j=next(j for j in r['joints'] if j['name']=='index_dip');m=j['mimic']
    assert targets['index_dip']==pytest.approx(float(m['multiplier'])*targets[m['joint']]+float(m['offset']))
    assert old==r


def test_proposals_reject_bounds_dependent_dofs_and_wrong_parent():
    r=reference();v=proposals(r,['index_pip'],.01)[0]
    for change in [dict(joint_deltas_rad={'index_dip':.01}),dict(joint_deltas_rad={'index_pip':.031}),
                   dict(parent_scene_sha256='wrong'),dict(wrist_translation_m=[.003,0,0]),dict(wrist_rotation_vector_deg=[3.,0,0])]:
        bad=dict(v,**change)
        with pytest.raises(ValueError):variant_targets(r,bad)
    for v in proposals(r,['thumb_mcp','index_pip'],.015,True):variant_targets(r,v)


def test_export_new_source_preserves_parent_and_moves_stale_energies(tmp_path):
    """Only an export wiring fixture, not a physical validation claim."""
    from scripts.export_refined_grasp import export
    from isaacsim_grasp.reference import digest
    parent=reference();v=proposals(parent,['index_pip'],.01)[0]
    r=copy.deepcopy(parent);r['derived_variant']=v;r['targets']=variant_targets(parent,v);r['qpos'].update(r['targets'])
    selected=tmp_path/'selected'/'a';selected.mkdir(parents=True)
    (selected.parent/'worker_plan.json').write_text(json.dumps(dict(reference=parent)))
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'reference.json').write_text(json.dumps(r))
    (bundle/'validation.json').write_text(json.dumps(dict(selected_run=str(selected),best_pass_count=6,baseline_pass_count=4,verified_scope='TEST FIXTURE ONLY')))
    result=export(bundle,tmp_path/'export')
    assert digest(result['parent_source'])==result['parent_source_sha256']
    a=np.load(result['result_file'],allow_pickle=True);old=np.load(result['parent_source'],allow_pickle=True)
    i=parent['candidate'];assert 'E_pen' not in a[i] and 'E_pen' in a[i]['parent_static_metrics']
    assert a[i]['scale']==old[i]['scale']
    assert a[i]['qpos']['index_pip']==pytest.approx(r['targets']['index_pip'])
    assert a[1]['qpos']==old[1]['qpos']
    with pytest.raises(FileExistsError):export(bundle,tmp_path/'export')
