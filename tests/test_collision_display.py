"""Ensure collision inspection renders hulls, with independent object visibility."""

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'grasp_generation'))
from utils.collision_display import prepare_mesh_groups, set_mesh_visibility, mesh_counts


class CollisionDisplayTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
          <mesh name="tetra" vertex="0 0 0 .02 0 0 0 .02 0 0 0 .02"
                face="0 2 1 0 1 3 1 2 3 0 3 2"/>
          </asset><worldbody>
          <body name="hand"><geom type="mesh" mesh="tetra"/>
            <geom type="mesh" mesh="tetra" contype="0" conaffinity="0" density="0"/></body>
          <body name="object" pos=".1 0 0"><freejoint/>
            <geom type="mesh" mesh="tetra"/>
            <geom type="mesh" mesh="tetra" contype="0" conaffinity="0" density="0"/>
          </body></worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def test_grouping_does_not_change_physics(self):
        masks = np.column_stack((self.model.geom_contype.copy(), self.model.geom_conaffinity.copy()))
        mass = self.model.body_mass.copy()
        groups = prepare_mesh_groups(self.model)
        self.assertTrue(all(n == 1 for n in mesh_counts(groups).values()))
        np.testing.assert_array_equal(masks, np.column_stack((self.model.geom_contype, self.model.geom_conaffinity)))
        np.testing.assert_array_equal(mass, self.model.body_mass)

    def test_separate_visibility_and_real_collision_hulls(self):
        prepare_mesh_groups(self.model)
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(self.model, maxgeom=20)
        camera = mujoco.MjvCamera()
        for hand, cup, expected in [('collision','collision',2), ('collision','none',1), ('none','collision',1)]:
            set_mesh_visibility(option, hand, cup)
            mujoco.mjv_updateScene(self.model,self.data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            self.assertEqual(scene.ngeom, expected)
            # MuJoCo mesh data IDs are 2*mesh_id for visual assets, +1 for hulls.
            self.assertTrue(all(scene.geoms[i].dataid % 2 == 1 for i in range(scene.ngeom)))
        set_mesh_visibility(option, 'visual', 'visual')
        mujoco.mjv_updateScene(self.model,self.data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        self.assertTrue(all(scene.geoms[i].dataid % 2 == 0 for i in range(scene.ngeom)))


if __name__ == '__main__':
    unittest.main()
