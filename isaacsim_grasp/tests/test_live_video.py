import sys,types
import numpy as np
import pytest
from isaacsim_grasp.live_video import LiveDynamicsVideo


@pytest.mark.parametrize('change',[None,'state','clock'])
def test_render_does_not_advance_physics_and_restores_setting(monkeypatch,change):
    values={'/app/player/playSimulations':True}
    settings=types.SimpleNamespace(get=values.get,set=lambda k,v:values.__setitem__(k,v))
    sub=types.ModuleType('carb.settings');sub.get_settings=lambda:settings
    carb=types.ModuleType('carb');carb.settings=sub
    monkeypatch.setitem(sys.modules,'carb',carb);monkeypatch.setitem(sys.modules,'carb.settings',sub)
    position=np.array([0.]);sim=types.SimpleNamespace(current_time=0.,render=lambda:None)
    def step(**kw):
        assert kw['delta_time']==0. and kw['pause_timeline'] is False
        assert values['/app/player/playSimulations'] is False
        if change=='state':position[0]=1.
        if change=='clock':sim.current_time=1.
    video=LiveDynamicsVideo.__new__(LiveDynamicsVideo)
    video.env=types.SimpleNamespace(state=lambda:(position.copy(),),sim=sim)
    video.rep=types.SimpleNamespace(orchestrator=types.SimpleNamespace(step=step))
    if change:
        with pytest.raises(RuntimeError,match='changed physical'):video.render_checked()
    else:video.render_checked()
    assert values['/app/player/playSimulations'] is True
