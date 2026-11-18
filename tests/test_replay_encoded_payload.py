import base64
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_l20_isaacsim_html import render_html,VENDOR


def test_robot_title_does_not_rewrite_encoded_tokens():
    # L20 is valid base64 text and must not be replaced by the visible robot name.
    encoded='AAAL20BBB+/=='
    page=render_html(encoded,'ShadowHand')
    assert f"new Blob([bytes('{encoded}')])" in page
    assert '<title>ShadowHand Isaac Sim' in page
    modules=re.findall('data:text/javascript;base64,([A-Za-z0-9+/=]+)',page)
    assert len(modules)==2
    assert base64.b64decode(modules[0])==(VENDOR/'three.module.min.js').read_bytes()
    assert base64.b64decode(modules[1])==(VENDOR/'OrbitControls.js').read_bytes()


def test_title_escapes_markup():
    page=render_html('AA==','hand<script>')
    assert '<title>hand&lt;script&gt;' in page
