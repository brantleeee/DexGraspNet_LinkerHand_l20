"""Official native mechanism survives import and rejects incompatible poses."""

import copy
import tempfile
import unittest

import mujoco
import numpy as np

from mjlab_grasp.shadowhand import OFFICIAL, build_shadow_template, prepare_shadow_grasp, verify_official_assets
from mjlab_grasp.candidates import ROOT
from mjlab_grasp.physics import Physics, compile_scene


class ShadowHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.code = "core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03"
        cls.scene, cls.metadata = build_shadow_template(cls.code, cls.temporary.name)
        cls.grasp = np.load(OFFICIAL / "data/dataset" / (cls.code + ".npy"), allow_pickle=True)[0]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_pinned_sources(self):
        self.assertEqual(verify_official_assets()["commit"], "bd1b13d7248729af117e1d46aaa6266b147a3c7b")

    def test_mechanics_and_initial_pose_preserved(self):
        original = copy.deepcopy(self.grasp)
        scene = prepare_shadow_grasp(self.scene, self.grasp, 3)
        model = compile_scene(scene, kinematic_wrist=True)
        self.assertEqual((model.nu, model.ntendon, model.nq, model.neq), (18, 4, 36, 0))
        np.testing.assert_allclose(model.actuator_gainprm[:, 0], 1)
        self.assertTrue(model.tendon_limited.all())
        self.assertEqual(self.grasp, original)
        physics = Physics(model, [self.grasp], "cpu")
        for name, value in self.grasp["qpos"].items():
            if name.startswith("robot0:"):
                self.assertAlmostEqual(physics.read("qpos")[0, model.joint(name).qposadr[0]].item(), value, places=6)
        passive = {"robot0:" + f + "J0" for f in ("FF", "MF", "RF", "LF")}
        driven = {model.joint(int(j)).name for j in model.actuator_trnid[:, 0]}
        self.assertFalse(passive & driven)
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            self.assertEqual(model.sensor(finger + "_force").dim[0], 96)
        # Official example violates its native tendon interval. This must be
        # measured as a length error, rather than reported as zero mimic error.
        self.assertGreater(physics.tendon_errors().item(), 0.001)
        self.assertEqual(physics.joint_errors()[1].item(), 0)
        q = physics.read("qpos").clone()
        for name in passive | {"robot0:" + f + "J1" for f in ("FF", "MF", "RF", "LF")}:
            q[:, model.joint(name).qposadr[0]] = 0
        physics.write("qpos", q)
        physics.forward()
        self.assertEqual(physics.tendon_errors().item(), 0)

    def test_l20_or_incomplete_candidate_rejected(self):
        grasp = copy.deepcopy(self.grasp)
        grasp["qpos"]["index_drive"] = grasp["qpos"].pop("robot0:FFJ3")
        with self.assertRaisesRegex(ValueError, "native ShadowHand"):
            prepare_shadow_grasp(self.scene, grasp, 3)

    def test_declared_defaults_are_explicit_opt_in(self):
        np.testing.assert_allclose(self.metadata["finger_damping"], 0)
        with tempfile.TemporaryDirectory() as output:
            _, metadata = build_shadow_template(self.code, output, "asset-defaults")
        np.testing.assert_allclose(metadata["finger_damping"], 0.1)
        np.testing.assert_allclose(metadata["finger_armature"], 0.001)
        np.testing.assert_allclose(metadata["native_actuator_gains"], 1)

    def test_local_jenga_mesh_keeps_official_shadowhand_mechanism(self):
        mesh = ROOT / "data/meshdata/2_of_Jenga_Classic_Game/coacd/decomposed.obj"
        with tempfile.TemporaryDirectory() as output:
            scene, metadata = build_shadow_template("2_of_Jenga_Classic_Game", output,
                                                     object_mesh=mesh)
            configured = prepare_shadow_grasp(scene, self.grasp, 3)
            model = compile_scene(configured, kinematic_wrist=True)
        self.assertEqual((model.nu, model.ntendon), (18, 4))
        self.assertEqual(metadata["object_convex_parts"], 6)
        self.assertEqual(metadata["object_mesh_file"], str(mesh.resolve()))
        self.assertEqual(len([g for g in range(model.ngeom)
                              if model.geom_bodyid[g] == model.body("object").id
                              and model.geom_contype[g]]), 6)


if __name__ == "__main__":
    unittest.main()
