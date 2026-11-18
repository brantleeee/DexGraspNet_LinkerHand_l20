import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import batch_l20_isaacsim as batch


def test_new_batch_executes_input_preparation(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['batch','--object','power-drill','--result-file','unused.npy','--output',str(tmp_path/'new')])
    def fake_run(cmd,**kwargs):
        assert Path(cmd[1]).name=='prepare_l20_isaacsim.py'
        assert cmd[cmd.index('--object')+1]=='power-drill'
        raise RuntimeError('preparation was executed')
    monkeypatch.setattr(batch.subprocess,'run',fake_run)
    with pytest.raises(RuntimeError,match='preparation was executed'):batch.main()
