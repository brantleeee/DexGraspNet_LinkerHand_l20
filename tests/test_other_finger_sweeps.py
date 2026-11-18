import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_fingers_mujoco_full_hand import motion_paths
from check_middle_mujoco_full_hand import project_mimic


def test_all_finger_paths_respect_limits_step_and_mimic():
    robot=ET.parse(Path(__file__).resolve().parents[1]/'grasp_generation/mjcf/linkerhand_l20_right.urdf').getroot()
    joints={j.get('name'):j for j in robot.findall('joint') if j.get('type')!='fixed'}
    for finger in ['index','ring','pinky','thumb']:
        names,paths=motion_paths(robot,finger,.01)
        assert len(paths)==(7 if finger=='thumb' else 5)
        for rows in paths.values():
            assert np.abs(np.diff(rows,axis=0)).max()<=.01000001
            for i,name in enumerate(names):
                lim=joints[name].find('limit');assert rows[:,i].min()>=float(lim.get('lower'))-1e-10
                assert rows[:,i].max()<=float(lim.get('upper'))+1e-10
            base={n:0. for n in joints};base.update(dict(zip(names,rows[-1])));q=project_mimic(robot,base)
            for n in [finger+'_dip']:
                lim=joints[n].find('limit');assert float(lim.get('lower'))-1e-10<=q[n]<=float(lim.get('upper'))+1e-10
