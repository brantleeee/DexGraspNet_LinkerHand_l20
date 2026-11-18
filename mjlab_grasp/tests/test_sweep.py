"""Candidate coverage and interrupted-run recovery without running simulation."""

import json
import fcntl
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from mjlab_grasp.sweep import completed_attempt, run_sequential, select_indices, worker_command, sweep_settings, settings_changes


class SweepTests(unittest.TestCase):
    def test_settings_changes_identifies_files_and_runtime(self):
        before = {"assets": {"/tmp/physics.py": "old", "/tmp/deleted": "hash"}, "runtime": {"python": "old"}}
        after = {"assets": {"/tmp/physics.py": "new", "/tmp/new": "hash"}, "runtime": {"python": "new"}}
        changes = settings_changes(before, after)
        self.assertEqual({(c["field"], c["kind"]) for c in changes}, {
            ("assets./tmp/physics.py", "modified"), ("assets./tmp/deleted", "removed"),
            ("assets./tmp/new", "added"), ("runtime.python", "modified")})
        self.assertEqual(settings_changes(before, before), [])

    def test_backend_report_accepts_zero_mocap_bodies(self):
        from mjlab_grasp.reporting import write_report
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            rows = []
            for backend in ["mjlab", "cpu"]:
                filename = f"{backend}.npz"
                np.savez(output/filename, initial_qpos=np.zeros(7), initial_qvel=np.zeros(6),
                    initial_ctrl=np.zeros(1), initial_mocap_pos=np.zeros((0,3)), initial_mocap_quat=np.zeros((0,4)))
                rows.append({"candidate": 0, "variant": "original", "backend": backend, "case": "lift",
                    "wrist_motion": "kinematic", "passed": False, "dynamics_pass": False, "trajectory": filename,
                    "max_relative_slip_mm": 10., "max_rotation_deg": 20., "peak_object_contact_depth_mm": 0.})
            (output/"results.json").write_text(json.dumps(rows))
            write_report(output)
            pairs = json.loads((output/"backend_comparison.json").read_text())
            self.assertEqual(pairs[0]["initial_state_max_abs_difference"]["initial_mocap_pos"], 0.)

    def test_all_means_every_candidate_without_energy_filter(self):
        data = [{"scale": 0.06, "energy": 100}, {"scale": 0.08}, {"scale": 0.06}]
        self.assertEqual(select_indices(data), [0, 1, 2])
        self.assertEqual(select_indices(data, scale=0.06), [0, 2])
        self.assertEqual(select_indices(data, [2, 0]), [2, 0])
        for indices, scale in [([3], None), ([0, 0], None), ([1], 0.06), (None, float("nan"))]:
            with self.assertRaises(ValueError):
                select_indices(data, indices, scale)

    def args(self, output):
        return SimpleNamespace(output=output, resume=False, indices=[0, 1, 2], compare_cpu=False,
            backend="cpu", cases=["gz-", "lift"], result_file=output.parent / "grasps.npy",
            required_fingers=["thumb", "index"], friction=3., timestep=0.001, protocol="standard",
            hold_seconds=2., refine_seconds=0.5, max_joint_delta=0.25, max_initial_gap=0.02, wrist_motion="kinematic",
            self_collision=False, skip_refinement=False)

    def fake_worker(self, command, **kwargs):
        index = int(command[command.index("--indices") + 1])
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        if index in self.error_indices:
            return SimpleNamespace(wait=lambda: 1)
        rows = []
        for case in ["gz-", "lift"]:
            artifact = f"{case}.npz"
            np.savez(output / artifact, time=[.01, .02], qpos=np.zeros((2,1)), qvel=np.zeros((2,1)), ctrl=np.zeros((2,1)),
                mocap_pos=np.zeros((2,0,3)), mocap_quat=np.zeros((2,0,4)), initial_qpos=np.zeros(1), initial_qvel=np.zeros(1),
                initial_ctrl=np.zeros(1), initial_mocap_pos=np.zeros((0,3)), initial_mocap_quat=np.zeros((0,4)))
            (output / "scene.xml").write_text("<mujoco/>")
            rows.append({"candidate": index, "variant": "original", "backend": "cpu", "case": case,
                "passed": False, "dynamics_pass": False, "trajectory": artifact, "scene": "scene.xml",
                "max_relative_slip_mm": 10., "max_rotation_deg": 20., "peak_object_contact_depth_mm": 0.})
        (output / "results.json").write_text(json.dumps(rows))
        (output / "asset_manifest.json").write_text('{}')
        (output / "summary.json").write_text(json.dumps({"completed": True, "trials": 2, "passed_trials": 0, "fully_passed": []}))
        return SimpleNamespace(wait=lambda: 0)

    def test_error_continues_and_resume_retries_only_unfinished_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            args = self.args(output)
            self.error_indices = {1}
            with patch("mjlab_grasp.sweep.sweep_settings", return_value={"fixed": True}), patch(
                "mjlab_grasp.sweep.subprocess.Popen", side_effect=self.fake_worker) as worker:
                self.assertEqual(run_sequential(args), 1)
                self.assertEqual(worker.call_count, 3)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["completed_candidates"], 2)
                self.assertEqual(summary["error_candidates"], 1)
                self.assertFalse(summary["completed"])
                self.assertTrue(summary["finished"])
                self.error_indices.clear()
                args.resume = True
                worker.reset_mock()
                self.assertEqual(run_sequential(args), 0)
                self.assertEqual(worker.call_count, 1)
            rows = json.loads((output / "results.json").read_text())
            self.assertEqual(len(rows), 6)
            self.assertEqual({r["candidate"] for r in rows}, {0, 1, 2})
            self.assertTrue((output / "candidate_001/attempt_0002/summary.json").exists())
            self.assertTrue(json.loads((output / "summary.json").read_text())["completed"])
            # A completed physical failure is retained, not rerun until it passes.
            self.assertEqual(sum(r["passed"] for r in rows), 0)
            self.assertTrue(all((output / r["trajectory"]).exists() for r in rows))

    def test_missing_artifact_and_partial_case_matrix_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp) / "run")
            parent = Path(tmp) / "candidate_000"
            parent.mkdir()
            attempt = parent / "attempt_0001"
            self.error_indices = set()
            self.fake_worker(worker_command(args, 0, attempt))
            self.assertIsNotNone(completed_attempt(parent, 0, args.cases, ["cpu"]))
            self.assertIsNone(completed_attempt(parent, 0, [*args.cases, "gx+"], ["cpu"]))
            (attempt / "lift.npz").unlink()
            self.assertIsNone(completed_attempt(parent, 0, args.cases, ["cpu"]))

    def test_resume_rejects_changed_settings_before_starting_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "sweep_settings.json").write_text('{"fixed": false}')
            args = self.args(output)
            args.resume = True
            with patch("mjlab_grasp.sweep.sweep_settings", return_value={"fixed": True}), patch(
                "mjlab_grasp.sweep.subprocess.Popen") as worker:
                with self.assertRaisesRegex(ValueError, "Resume rejected"):
                    run_sequential(args)
                worker.assert_not_called()

    def test_corrupt_trajectory_and_inconsistent_summary_are_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)/"candidate_000"
            parent.mkdir()
            args = self.args(Path(tmp)/"run")
            attempt = parent/"attempt_0001"
            self.error_indices = set()
            self.fake_worker(worker_command(args, 0, attempt))
            summary = json.loads((attempt/"summary.json").read_text())
            summary["fully_passed"] = [{"candidate":0,"variant":"original","backend":"cpu"}]
            (attempt/"summary.json").write_text(json.dumps(summary))
            self.assertIsNone(completed_attempt(parent, 0, args.cases, ["cpu"]))
            summary["fully_passed"] = []
            (attempt/"summary.json").write_text(json.dumps(summary))
            (attempt/"lift.npz").write_bytes(b"damaged archive")
            self.assertIsNone(completed_attempt(parent, 0, args.cases, ["cpu"]))

    def test_same_output_cannot_be_used_by_two_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = self.args(output)
            args.resume = True
            with (output/".sweep.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "Another process"):
                    run_sequential(args)

    def test_mutation_during_worker_prevents_mixed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp)/"run")
            revision = [1]
            self.error_indices = set()
            def launch(command, **kwargs):
                result = self.fake_worker(command, **kwargs)
                revision[0] = 2
                return result
            with patch("mjlab_grasp.sweep.sweep_settings", side_effect=lambda args: {"revision":revision[0]}), patch(
                "mjlab_grasp.sweep.subprocess.Popen", side_effect=launch) as worker:
                with self.assertRaisesRegex(RuntimeError, "changed during"):
                    run_sequential(args)
                self.assertEqual(worker.call_count, 1)
            self.assertFalse(json.loads((args.output/"summary.json").read_text())["completed"])
            self.assertTrue((args.output/"run_error.json").exists())
            changes = json.loads((args.output/"settings_changes.json").read_text())["changes"]
            self.assertEqual(changes, [{"field": "revision", "kind": "modified", "previous": 1, "current": 2}])

    def test_contact_asset_changes_are_fingerprinted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contact = root/"grasp_generation/mjcf/contact_points.json"
            contact.parent.mkdir(parents=True)
            contact.write_text('{}')
            args = self.args(root/"run")
            args.indices = [0]
            np.save(args.result_file, [{"scale":.06}], allow_pickle=True)
            with patch("mjlab_grasp.sweep.ROOT", root):
                before = sweep_settings(args)
                contact.write_text('{"changed":true}')
                after = sweep_settings(args)
            self.assertNotEqual(before, after)

    def test_report_error_does_not_publish_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp)/"run")
            args.indices = [0]
            self.error_indices = set()
            with patch("mjlab_grasp.sweep.sweep_settings", return_value={"fixed":True}), patch(
                "mjlab_grasp.sweep.subprocess.Popen", side_effect=self.fake_worker), patch(
                "mjlab_grasp.sweep.write_report", side_effect=OSError("report failed")):
                with self.assertRaisesRegex(OSError, "report failed"):
                    run_sequential(args)
            summary = json.loads((args.output/"summary.json").read_text())
            self.assertFalse(summary["completed"])
            self.assertFalse(summary["report_completed"])

    def test_worker_launch_error_is_logged_and_remaining_candidates_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp)/"run")
            self.error_indices = set()
            def launch(command, **kwargs):
                if command[command.index("--indices")+1] == "1":
                    raise OSError("launch failed")
                return self.fake_worker(command, **kwargs)
            with patch("mjlab_grasp.sweep.sweep_settings", return_value={"fixed":True}), patch(
                "mjlab_grasp.sweep.subprocess.Popen", side_effect=launch):
                self.assertEqual(run_sequential(args), 1)
            summary = json.loads((args.output/"summary.json").read_text())
            self.assertEqual(summary["completed_candidates"], 2)
            self.assertEqual(summary["error_candidates"], 1)

    def test_retry_after_launch_error_preserves_the_previous_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp)/"run")
            args.indices = [0]
            self.error_indices = set()
            with patch("mjlab_grasp.sweep.sweep_settings", return_value={"fixed":True}):
                with patch("mjlab_grasp.sweep.subprocess.Popen", side_effect=OSError("first launch failed")):
                    self.assertEqual(run_sequential(args), 1)
                first = args.output/"candidate_000/attempt_0001.log"
                original = first.read_text()
                args.resume = True
                with patch("mjlab_grasp.sweep.subprocess.Popen", side_effect=self.fake_worker):
                    self.assertEqual(run_sequential(args), 0)
                self.assertEqual(first.read_text(), original)
                self.assertTrue((args.output/"candidate_000/attempt_0002/summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
