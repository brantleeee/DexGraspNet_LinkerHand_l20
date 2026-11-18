import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from open_l20_grasp_results import inspect


def test_running_progress_does_not_require_html(tmp_path):
    run=tmp_path/'run';run.with_suffix('.job.json').write_text(json.dumps({'phase':'generating'}))
    run.with_suffix('.log').write_text('optimizing: 10% 600/6000 [01:00<09:00]\roptimizing: 11% 660/6000 [01:06<08:54]')
    state=inspect(run)
    assert state['phase']=='generating' and state['progress']==(660,6000)
    assert not state['target'].exists()


def test_completed_job_uses_saved_export_path(tmp_path):
    run=tmp_path/'run';target=tmp_path/'export/index.html';target.parent.mkdir();target.write_text('ready')
    run.with_suffix('.job.json').write_text(json.dumps({'phase':'complete','html':str(target)}))
    assert inspect(run)['target']==target
