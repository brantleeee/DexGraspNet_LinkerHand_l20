"""Prepare a new replay model with eight root pairs disabled; never reuse old contact labels."""
import argparse,json,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,mujoco
from l20_root_collision_policy import apply_root_policy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--policy',choices=['angle-only','all-roots-angle-only'],default='all-roots-angle-only');a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    scene=ET.parse(a.source/'scene.xml').getroot();disabled=apply_root_policy(scene,a.policy)
    old=mujoco.MjModel.from_xml_path(str(a.source/'scene.xml'));model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
    ignored={frozenset(p) for p in disabled}
    assert model.npair==231-len(disabled) and model.nexclude==len(disabled) and model.ngeom==22 and model.neq==5
    np.testing.assert_array_equal(model.jnt_range,old.jnt_range);np.testing.assert_array_equal(model.eq_data,old.eq_data)
    actual={frozenset((model.body(int(model.geom_bodyid[g1])).name,model.body(int(model.geom_bodyid[g2])).name)) for g1,g2 in zip(model.pair_geom1,model.pair_geom2)}
    assert not actual&ignored
    original=json.loads((a.source/'results.json').read_text());cases=[];checks=[];data=mujoco.MjData(model)
    for c in original['cases']:
        trace=(a.source/c['trajectory']).resolve();q=np.load(trace)['qpos']
        cases.append({**{k:c[k] for k in ['name','context','path','frames']},'finger':c.get('finger','middle'),'trajectory':str(trace)})
        for frame in [0,len(q)-1]:
            data.qpos[:]=q[frame];mujoco.mj_forward(model,data)
            for contact in data.contact:
                pair=frozenset((model.body(int(model.geom_bodyid[contact.geom1])).name,model.body(int(model.geom_bodyid[contact.geom2])).name));assert pair not in ignored
            checks.append(dict(case=c['name'],frame=frame,contacts=data.ncon))
    a.output.mkdir(parents=True);(a.output/'scene.xml').write_text(ET.tostring(scene,encoding='unicode'))
    (a.output/'results.json').write_text(json.dumps(dict(complete=True,mode='replay model prepared; old contact summaries intentionally omitted; endpoint checks only',cases=cases),indent=2)+'\n')
    (a.output/'collision_policy.json').write_text(json.dumps(dict(policy=a.policy,disabled_pairs=disabled,remaining_pairs=model.npair,links=22,angle_limits_unchanged=True,mimic_unchanged=True,endpoint_checks=checks,source_scene=str((a.source/'scene.xml').resolve()),source_sha256=hashlib.sha256((a.source/'scene.xml').read_bytes()).hexdigest()),indent=2)+'\n')
    print(a.output.resolve(),str(model.npair)+' pairs, limits/mimic preserved, endpoint checks',len(checks))


if __name__=='__main__':main()
