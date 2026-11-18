import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from mjlab_grasp.sdf import convert_xml, check_reference_compatibility


class SdfReferenceTests(unittest.TestCase):
    def test_collision_only_and_inertia_preserved(self):
        root = ET.fromstring('''<mujoco><asset><mesh name="m" vertex="0 0 0  .01 0 0  0 .01 0  0 0 .01"/></asset>
        <worldbody><body name="object"><freejoint/>
        <geom type="mesh" mesh="m" density="500"/>
        <geom type="mesh" mesh="m" contype="0" conaffinity="0" density="0"/>
        </body></worldbody></mujoco>''')
        old = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        converted, new, count = convert_xml(root)
        self.assertEqual(count, 1)
        self.assertEqual(list(new.geom_type), [mujoco.mjtGeom.mjGEOM_SDF, mujoco.mjtGeom.mjGEOM_MESH])
        np.testing.assert_allclose(old.body_mass, new.body_mass)
        np.testing.assert_allclose(old.body_inertia, new.body_inertia)
        self.assertIsNone(root.find('.//inertial'))
        self.assertIsNotNone(converted.find('.//inertial'))

    def test_legacy_checkpoint_cannot_load_sdf(self):
        for current, saved in [({'collision_representation':'mesh_sdf'}, {}), ({}, {'collision_representation':'mesh_sdf'})]:
            with self.assertRaises(ValueError):
                check_reference_compatibility(current, saved)

    def test_sdf_requires_identical_assets(self):
        with self.assertRaises(ValueError):
            check_reference_compatibility({'collision_representation':'mesh_sdf','asset_sha256':{'hand':'a'}},
                                          {'collision_representation':'mesh_sdf','asset_sha256':{'hand':'b'}})
        check_reference_compatibility({}, {})


if __name__ == '__main__':
    unittest.main()
