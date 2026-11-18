"""Physical invariants for candidate refinement, independent of reward shaping."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from mjlab_grasp.candidates import FINGERS, initialize_data
from mjlab_grasp.metrics import relative_pose, pose_error, advance_hold, advance_hold_steps, required_hold_steps
from mjlab_grasp.physics import Physics, compile_scene, independent_transmission, point_jacobian
from mjlab_grasp.validation import ValidationConfig, validate
from mjlab_grasp.refinement import refine, RefinementConfig


def fixture(object_pos="0.19 0 0"):
    scene = ET.fromstring(f'<mujoco><compiler angle="radian"/><option gravity="0 0 0" integrator="implicitfast"/>'
        '<worldbody><body name="wrist" mocap="true"/></worldbody><actuator/><equality/></mujoco>')
    wrist = scene.find("worldbody/body")
    qpos = {}
    for i, finger in enumerate(FINGERS):
        body = ET.SubElement(wrist, "body", name=finger + "_root", pos=f"0 {i} 0")
        ET.SubElement(body, "joint", name=finger + "_drive", axis="0 0 1", range="-1 1", damping="0.01")
        ET.SubElement(body, "geom", type="sphere", size="0.05", pos="0.1 0 0", mass="0.1")
        ET.SubElement(scene.find("actuator"), "general", joint=finger + "_drive", gainprm="5", biastype="affine",
            biasprm="0 -5 -0.05", ctrlrange="-1 1", forcerange="-1 1")
        qpos[finger + "_drive"] = 0.1
        if finger == "index":
            child = ET.SubElement(body, "body", name="index_tip", pos="0.1 0 0")
            ET.SubElement(child, "joint", name="index_passive", axis="0 0 1", range="-1 1")
            ET.SubElement(child, "geom", type="sphere", size="0.02", pos="0.1 0 0", mass="0.02")
            ET.SubElement(scene.find("equality"), "joint", joint1="index_passive", joint2="index_drive", polycoef="0 0.5 0 0 0")
            qpos["index_passive"] = 0.05
    obj = ET.SubElement(scene.find("worldbody"), "body", name="object", pos=object_pos)
    ET.SubElement(obj, "freejoint", name="object_free")
    ET.SubElement(obj, "geom", type="sphere", size="0.05", mass="0.1")
    return scene, {"scale": 0.06, "qpos": qpos}


class RefinementValidationTests(unittest.TestCase):
    def test_post_lift_hold_does_not_include_the_last_lifting_step(self):
        # A supported box is a physical positive control for timing, not a
        # dexterous-grasp benchmark. Only thumb load is required in this fixture.
        scene, grasp = fixture("0.1 0 0.04")
        platform = scene.find("worldbody/body[@name='wrist']/body[@name='thumb_root']/geom")
        platform.set("type", "box")
        platform.set("size", "0.1 0.1 0.01")
        platform.set("pos", "0.1 0 0")
        obj = scene.find("worldbody/body[@name='object']/geom")
        obj.set("type", "box")
        obj.set("size", "0.03 0.03 0.03")
        grasp["qpos"]["thumb_drive"] = 0.
        for duration, count, passed in [(.999, 999, False), (1., 1000, True)]:
            with self.subTest(hold_seconds=duration), tempfile.TemporaryDirectory() as tmp:
                row = validate(scene, [grasp], ["supported_box"], "lift", Path(tmp), "cpu",
                               ValidationConfig(hold_seconds=duration), ("thumb",))[0]
                self.assertEqual(row["final_stable_steps"], count)
                self.assertEqual(row["dynamics_pass"], passed)

    def test_exact_one_second_stability_window_has_no_float_accumulation_error(self):
        count = torch.zeros(1, dtype=torch.int64)
        required = required_hold_steps(1., .001)
        for _ in range(999):
            count = advance_hold_steps(count, torch.tensor([True]))
        self.assertFalse(bool(count[0] >= required))
        count = advance_hold_steps(count, torch.tensor([True]))
        self.assertTrue(bool(count[0] >= required))
        self.assertEqual(required_hold_steps(1., .003), 334)
        self.assertEqual(advance_hold_steps(count, torch.tensor([False]))[0].item(), 0)
    def test_scene_snapshot_matches_runtime_options_and_can_be_recompiled(self):
        scene, grasp = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"scene.xml"
            model = compile_scene(scene, kinematic_wrist=True, xml_path=path)
            physics = Physics(model, [grasp], "cpu", timestep=0.00025)
            physics.save_scene(path)
            replay = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual(replay.opt.timestep, 0.00025)
            self.assertEqual(replay.opt.ccd_iterations, model.opt.ccd_iterations)
            self.assertEqual(replay.opt.solver, model.opt.solver)
            recompiled = compile_scene(ET.parse(path).getroot(), kinematic_wrist=True)
            self.assertEqual(recompiled.nq, model.nq)
            self.assertEqual(recompiled.nsensor, model.nsensor)

    def test_refinement_settling_checks_passive_joint_speed(self):
        scene, grasp = fixture("0 0 -3")
        class MovingPassiveJoint(Physics):
            def step(self):
                super().step()
                adr = self.model.jnt_dofadr[self.model.joint("index_passive").id]
                for data in self.cpu:
                    data.qvel[adr] = 10.
                    mujoco.mj_forward(self.model, data)
        model = compile_scene(scene)
        data = mujoco.MjData(model)
        from mjlab_grasp.candidates import initialize_data
        data.qpos[:] = initialize_data(model, [grasp])[0][0]
        mujoco.mj_forward(model, data)
        bid = model.body("thumb_root").id
        point = data.xpos[bid] + data.xmat[bid].reshape(3,3) @ np.array([0.1,0,0])
        contacts = [{"body":"thumb_root", "point_local":[0.1,0,0], "target_world":point.tolist()}]
        with patch("mjlab_grasp.refinement.Physics", MovingPassiveJoint):
            _, report = refine(scene, grasp, contacts, "cpu", RefinementConfig(seconds=.002))
        self.assertGreaterEqual(report["trace"][0][2], 10.)
        self.assertGreaterEqual(report["final_finger_speed_rad_s"], 10.)
        self.assertFalse(report["target_converged"])

    def test_invalid_physics_configuration_is_rejected(self):
        for kwargs in [{"gravity_ramp_seconds":0.}, {"timestep":float("nan")}, {"protocol":"typo"}]:
            with self.assertRaises(ValueError):
                ValidationConfig(**kwargs)
        for kwargs in [{"seconds":0.}, {"margin":-.001}]:
            with self.assertRaises(ValueError):
                RefinementConfig(**kwargs)

    def test_prescribed_wrist_velocity_is_visible_to_contact_solver(self):
        scene, grasp = fixture("0 0 -3")
        config = ValidationConfig(settle_seconds=0.01, gravity_ramp_seconds=0.01,
            prelift_hold_seconds=0.01, lift_seconds=0.1, hold_seconds=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            row = validate(scene, [grasp], ["original"], "lift", Path(tmp), "cpu", config)[0]
            with np.load(Path(tmp)/row["trajectory"]) as trajectory:
                self.assertGreater(trajectory["wrist_solver_velocity"][:, 2].max(), 1.0)
                model = mujoco.MjModel.from_xml_path(str(Path(tmp)/row["scene"]))
                obj_vadr = model.jnt_dofadr[model.joint("object_free").id]
                np.testing.assert_array_equal(trajectory["qfrc_applied"][:, obj_vadr:obj_vadr+6], 0.)
            self.assertEqual(row["wrist_motion"], "kinematic")

    def test_moving_friction_platform_positive_control(self):
        from scripts.probe_mocap_friction import trial
        with tempfile.TemporaryDirectory() as tmp:
            mocap = trial("mocap", Path(tmp))
            prescribed = trial("slide_with_velocity", Path(tmp))
            self.assertLess(mocap["peak_solver_platform_velocity_m_s"], 1e-6)
            self.assertGreater(mocap["final_relative_slip_m"], 0.09)
            self.assertLess(prescribed["final_relative_slip_m"], 1e-4)

    def test_nearby_link_cannot_accept_an_unconverged_contact_point(self):
        scene, grasp = fixture("0.199 0 0")
        contacts = [{"body": "thumb_root", "point_local": [0.1, 0., 0.], "target_world": [0.149, 0., 0.]}]
        _, report = refine(scene, grasp, contacts, "cpu", RefinementConfig(seconds=0.001))
        self.assertTrue(report["geometry"]["passed"])
        self.assertGreater(report["final_target_error_m"], 0.002)
        self.assertGreater(report["exported_target_error_m"], 0.002)
        self.assertFalse(report["passed"])
        self.assertIn("target_convergence", report["failure_reasons"])
        json.dumps(report, allow_nan=False)

    def test_refinement_accepts_serialized_free_joint(self):
        scene, _ = fixture()
        model = compile_scene(scene)
        spec = mujoco.MjSpec.from_string(ET.tostring(scene, encoding="unicode"))
        serialized = ET.fromstring(spec.to_xml())
        pinned = compile_scene(serialized, fixed_object=True)
        self.assertEqual(pinned.nv + 6, model.nv)

    def test_each_world_restores_its_own_generated_wrist(self):
        scene, grasp = fixture()
        grasp["qpos"].update({"WRJTx": 0., "WRJTy": 0., "WRJTz": 0., "WRJRx": 0., "WRJRy": 0., "WRJRz": 0.})
        second = copy.deepcopy(grasp)
        second["qpos"].update({"WRJTx": 0.3, "WRJRz": np.pi / 2})
        physics = Physics(compile_scene(scene), [grasp, second], "cpu")
        root = physics.model.body("wrist").id
        np.testing.assert_allclose(physics.read("xpos")[:, root], [[0, 0, 0], [0.3, 0, 0]], atol=1e-7)
        np.testing.assert_allclose(physics.read("xmat")[1, root].reshape(3, 3),
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-7)

    def test_jacobian_matches_mujoco_and_passive_virtual_work(self):
        scene, grasp = fixture()
        model = compile_scene(scene, fixed_object=True, torque_control=True)
        physics = Physics(model, [grasp, grasp], "cpu")
        local = [[0.08, 0.01, 0.0]]
        bid = model.body("index_tip").id
        points, jac = point_jacobian(physics, [bid], local)
        expected = np.zeros((3, model.nv))
        mujoco.mj_jac(model, physics.cpu[0], expected, None, points[0, 0].numpy().astype(float), bid)
        np.testing.assert_allclose(jac[0, 0].numpy(), expected, atol=2e-7)
        matrix, _, _ = independent_transmission(model)
        force = np.array([0.3, 0.2, 0.1])
        du = np.arange(model.nu) * 0.03
        self.assertAlmostEqual(float(force @ expected @ matrix @ du), float((matrix.T @ expected.T @ force) @ du))
        parent = list(model.actuator_trnid[:, 0]).index(model.joint("index_drive").id)
        self.assertEqual(matrix[model.jnt_dofadr[model.joint("index_passive").id], parent], 0.5)

    def test_refinement_model_does_not_mutate_free_evaluation_scene(self):
        scene, _ = fixture()
        original_xml = ET.tostring(scene)
        pinned = compile_scene(scene, fixed_object=True, torque_control=True, margin=0.001)
        free = compile_scene(scene)
        self.assertEqual(pinned.nq + 7, free.nq)
        self.assertEqual(pinned.nv + 6, free.nv)
        self.assertEqual(ET.tostring(scene), original_xml)
        self.assertTrue(np.all(free.geom_margin == 0))
        np.testing.assert_allclose(free.actuator_biasprm[:, 1:3], [[-5, -0.05]] * free.nu)

    def test_native_contact_normal_load_matches_cpu_contact_force(self):
        scene, grasp = fixture()
        physics = Physics(compile_scene(scene), [grasp], "cpu")
        loads, counts, _, _ = physics.contacts()
        total = 0.0
        wrench = np.zeros(6)
        data = physics.cpu[0]
        oid = physics.model.body("object").id
        for ci, contact in enumerate(data.contact):
            if oid in physics.model.geom_bodyid[[contact.geom1, contact.geom2]]:
                mujoco.mj_contactForce(physics.model, data, ci, wrench)
                total += max(0., wrench[0])
        self.assertGreater(total, 0)
        self.assertGreater(counts.sum().item(), 0)
        self.assertAlmostEqual(loads.sum().item(), total, places=4)

    def test_world_transform_invariance_and_hold_resets_after_failure(self):
        identity = torch.eye(3)[None]
        hand = torch.tensor([[0., 0., 0.]])
        obj = torch.tensor([[0.1, 0., 0.]])
        initial, rotation = relative_pose(hand, identity, obj, identity)
        turn = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        offset = torch.tensor([[1., 2., 3.]])
        current, rot = relative_pose(offset, turn, (turn @ obj[..., None]).squeeze(-1) + offset, turn)
        slip, angle = pose_error(current, rot, initial, rotation)
        self.assertLess(slip.item(), 1e-6)
        self.assertEqual(angle.item(), 0)
        hold = advance_hold(torch.tensor([1.2, 0.2]), torch.tensor([False, True]), 0.01)
        torch.testing.assert_close(hold, torch.tensor([0., 0.21]))

    def test_free_fall_cannot_pass_and_batched_states_are_independent(self):
        scene, grasp = fixture("0 0 -3")
        second = copy.deepcopy(grasp)
        second["qpos"]["thumb_drive"] = -0.2
        physics = Physics(compile_scene(scene), [grasp, second], "cpu", gravity=(0, 0, -9.81))
        before = physics.read("qpos").clone()
        physics.step()
        self.assertFalse(torch.equal(before[0], before[1]))
        oid = physics.model.joint("object_free").qposadr[0]
        self.assertLess(physics.read("qpos")[0, oid+2].item(), before[0, oid+2].item())
        config = ValidationConfig(hold_seconds=0.02, settle_seconds=0.01, gravity_ramp_seconds=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            result = validate(scene, [grasp, second], ["a", "b"], "gz-", Path(tmp), "cpu", config)
            self.assertTrue(all(not r["dynamics_pass"] for r in result))
            trajectory = np.load(Path(tmp) / result[0]["trajectory"])
            self.assertIn("qvel", trajectory)
            self.assertLess(result[0]["final_height_gain_m"], 0)


if __name__ == "__main__":
    unittest.main()
