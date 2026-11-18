import copy,json
from pathlib import Path
import pytest
from scripts.compare_isaacsim_margin import compare


def test_same_saved_samples_allow_continuous_margin_comparison_and_reject_changes(tmp_path):
    source=Path('data/experiments/residual_v2_paired_0p5mm_20261105_153925')
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    for name in ['evaluation.json','perturbations.json','config.json','reference_snapshot.json']:
        text=(source/name).read_text();(old/name).write_text(text);(new/name).write_text(text)
    evaluation=json.loads((new/'evaluation.json').read_text())
    for row in evaluation['runs']['policy']:row['max_slip_m']*=.5;row['max_angle_deg']*=.5
    (new/'evaluation.json').write_text(json.dumps(evaluation))
    c=compare(new,old)
    assert c['metrics']['policy']['max_slip_m']['better']==30
    assert c['metrics']['policy']['current_pass']==c['metrics']['policy']['previous_pass']==30
    assert (new/'MARGIN_COMPARISON.md').exists()
    samples=json.loads((new/'perturbations.json').read_text());samples['offsets_m'][0][0][0]=0
    (new/'perturbations.json').write_text(json.dumps(samples))
    with pytest.raises(ValueError):compare(new,old)
