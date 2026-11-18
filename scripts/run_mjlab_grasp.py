"""Smoke-test, train, evaluate, or view the static-reference mjlab residual task."""

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/l20-mjlab-cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from mjlab_grasp.rewards import JOINT_SPEED_LIMIT, joint_speed_ok
from mjlab_grasp.task import (
    DEFAULT_REFERENCE,
    make_agent,
    make_env,
    relative_error,
    rotation_error,
    stable_success,
    strict_success,
)
from mjlab_grasp.viewer import GraspPolicyViewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["smoke", "train", "evaluate", "view"], default="smoke"
    )
    parser.add_argument(
        "--task-version",
        choices=["v1", "v2", "v3", "v4", "v5"],
        help="Default: v4 for new runs, checkpoint version for evaluation/resume",
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--actuator-damping", type=float,
        help="For evaluation/view: override joint position actuator damping (Nm s/rad)",
    )
    parser.add_argument(
        "--hold-target-rate", type=float,
        help="For v4 evaluation/view: cap target changes after 1.5 s, in rad/s",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Train v3 from a v1 actor; reset critic, optimizer and iteration",
    )
    parser.add_argument(
        "--view-seconds",
        type=float,
        help="Automatically close viewer after this duration",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/experiments/mjlab_grasp_smoke")
    )
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1 or args.steps < 1:
        parser.error("Environment, iteration and step counts must be positive")
    if args.mode in ("evaluate", "view") and args.checkpoint is None:
        parser.error("Evaluation/view requires --checkpoint")
    if args.warm_start and (args.mode != "train" or args.checkpoint is None):
        parser.error("--warm-start requires training with a checkpoint")
    if args.hold_target_rate is not None and (
        args.mode not in ("evaluate", "view") or
        not 0 < args.hold_target_rate <= 1.0
    ):
        parser.error("--hold-target-rate requires evaluate/view and a rate in (0, 1]")
    if args.actuator_damping is not None and (
        args.mode not in ("evaluate", "view") or
        not 0 < args.actuator_damping <= 1.0
    ):
        parser.error("--actuator-damping requires evaluate/view and a value in (0, 1]")
    if not torch.cuda.is_available():
        parser.error("MuJoCo-Warp task requires an accessible CUDA GPU")
    if args.output.exists():
        parser.error("Output already exists; select a new --output directory")
    args.output.mkdir(parents=True)
    reference = json.loads((args.reference / "reference.json").read_text())
    if args.checkpoint:
        from mjlab_grasp.sdf import check_reference_compatibility
        snapshot = args.checkpoint.parent / "reference_snapshot.json"
        saved_reference = json.loads(snapshot.read_text()) if snapshot.exists() else {}
        check_reference_compatibility(reference, saved_reference)
    (args.output / "reference_snapshot.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    saved_task = None
    if args.checkpoint:
        task_path = args.checkpoint.parent / "task_snapshot.json"
        saved_task = (
            json.loads(task_path.read_text())["version"] if task_path.exists() else "v1"
        )
    version = args.task_version or saved_task or "v4"
    warm_start = args.warm_start and saved_task == "v1" and version == "v3"
    if args.warm_start and not warm_start:
        parser.error("Warm start supports only a v1 actor to v3")
    if saved_task is not None and version != saved_task and not warm_start:
        parser.error(
            "Task version changed; start a new policy instead of loading incompatible checkpoint"
        )
    if args.hold_target_rate is not None and version != "v4":
        parser.error("--hold-target-rate currently supports v4 checkpoints only")
    (args.output / "task_snapshot.json").write_text(
        json.dumps(
            {
                "version": version,
                "warm_start_checkpoint": str(args.checkpoint) if warm_start else None,
            },
            indent=2,
        )
        + "\n"
    )
    cfg = make_env(
        args.reference, 1 if args.mode == "view" else args.num_envs,
        version=version, actuator_damping=args.actuator_damping,
    )
    if version in ("v3", "v4", "v5"):
        from mjlab_grasp import rewards as grasp_rewards

        shutil.copyfile(grasp_rewards.__file__, args.output / "rewards_used.py")
        (args.output / "reward_weights.json").write_text(
            json.dumps({n: t.weight for n, t in cfg.rewards.items()}, indent=2) + "\n"
        )
    if args.mode == "evaluate":
        cfg.auto_reset = False
    cfg.seed = 42
    torch.manual_seed(42)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        if reference.get("collision_representation") == "mesh_sdf":
            import mujoco
            model = env.sim.mj_model
            active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
            sdf_count = int(((model.geom_type == mujoco.mjtGeom.mjGEOM_SDF) & active).sum())
            convex_count = int(((model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & active).sum())
            if sdf_count != 23 or convex_count:
                raise ValueError(f"Expected 23 SDF collision geoms, got {sdf_count} SDF/{convex_count} mesh")
            (args.output / "sdf_runtime_audit.json").write_text(json.dumps({
                "backend": "mjlab/MuJoCo-Warp", "sdf_collision_geoms": sdf_count,
                "convex_mesh_collision_geoms": convex_count,
                "joint_equalities": int((model.eq_type == mujoco.mjtEq.mjEQ_JOINT).sum()),
                "mujoco_version": mujoco.__version__,
                "note": "Loaded GPU model; inspect smoke.json for finite-step completion",
            }, indent=2) + "\n")
        env.reset()
        term = env.action_manager.get_term("residual")
        assert term.action_dim == 16 and not any(
            n.endswith("_dip") for n in term.target_names
        )
        zeros = torch.zeros((env.num_envs, 16), device=env.device)
        env.action_manager.process_action(zeros)
        env.action_manager.apply_action()
        expected = env.scene["hand"].data.default_joint_pos[:, term.target_ids]
        actual = env.scene["hand"].data.joint_pos_target[:, term.target_ids]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        expected_from_file = torch.tensor(
            [
                cfg.scene.entities["hand"].init_state.joint_pos[n]
                for n in term.target_names
            ],
            device=env.device,
        )
        torch.testing.assert_close(
            expected, expected_from_file.expand_as(expected), atol=1e-6, rtol=0
        )
        target_scales = (
            torch.tensor(
                [cfg.actions["residual"].scale[n] for n in term.target_names],
                device=env.device,
            )
            if version in ("v4", "v5")
            else torch.full((16,), 0.2, device=env.device)
        )
        for raw in [-2.0, 2.0]:
            previous = term._processed_actions.clone()
            env.action_manager.process_action(torch.full_like(zeros, raw))
            env.action_manager.apply_action()
            target = env.scene["hand"].data.joint_pos_target[:, term.target_ids]
            assert ((target - expected).abs() <= target_scales + 1e-6).all()
            if version == "v4":
                for j, name in enumerate(term.target_names):
                    if name.startswith("thumb_"):
                        bound = reference["drive_limits"][name][0 if raw < 0 else 1]
                        torch.testing.assert_close(
                            target[:, j],
                            torch.full_like(target[:, j], bound),
                            atol=1e-6,
                            rtol=0,
                        )
            if version == "v2":
                assert (target - previous).abs().max() <= env.step_dt + 1e-6
            if version == "v5":
                limit = env.step_dt * env.action_manager.get_term("residual").capture_rate_rad_s
                assert (target - previous).abs().max() <= limit + 1e-6
        env.reset()
        if version == "v2":
            torch.testing.assert_close(
                term._processed_actions, expected, atol=1e-6, rtol=0
            )
        (args.output / "zero_reference_check.json").write_text(
            json.dumps(
                {
                    "scale": reference["scale"],
                    "source_group_index": reference["source_index"],
                    "independent_actions": 16,
                    "mimic_joints": 5,
                    "zero_action_matches_static_targets": version not in ("v4", "v5"),
                    "zero_action_matches_initial_targets": True,
                    "thumb_target_mode": "absolute feasible joint range"
                    if version in ("v4", "v5")
                    else "static residual",
                    "action_residual_bound_rad": None if version == "v4" else 0.2,
                    "non_thumb_residual_bound_rad": 0.2,
                    "task_version": version,
                    "target_rate_limit_rad_s": (
                        {"capture": 0.8, "hold": 0.2} if version == "v5"
                        else 1.0 if version == "v2" else None
                    ),
                    "drive_order": term.target_names,
                    "reference_type": "thumb: feasible range midpoint; other fingers: static offset"
                    if version in ("v4", "v5")
                    else "fixed static offset, not an increment from the current state",
                },
                indent=2,
            )
            + "\n"
        )
        wrapper = RslRlVecEnvWrapper(
            env, clip_actions=None if version in ("v2", "v3", "v4", "v5") else 1.0
        )
        if args.mode == "smoke":
            reward_sum = 0.0
            for _ in range(args.steps):
                obs, reward, _done, _info = wrapper.step(zeros)
                assert torch.isfinite(reward).all()
                for value in obs.values():
                    assert torch.isfinite(value).all()
                reward_sum += reward.mean().item()
            (args.output / "smoke.json").write_text(
                json.dumps(
                    {
                        "steps": args.steps,
                        "num_envs": env.num_envs,
                        "finite_rollout": True,
                        "zero_policy_mean_reward_sum": reward_sum,
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            agent = make_agent(version)
            agent.max_iterations = args.iterations
            agent.save_interval = max(1, min(100, args.iterations))
            agent_cfg = asdict(agent)
            (args.output / "agent.json").write_text(
                json.dumps(agent_cfg, indent=2) + "\n"
            )
            runner = MjlabOnPolicyRunner(wrapper, agent_cfg, str(args.output), "cuda:0")
            if args.checkpoint:
                saved = args.checkpoint.parent / "reference_snapshot.json"
                if not saved.exists() or json.loads(saved.read_text()) != reference:
                    parser.error(
                        "Checkpoint reference does not match this scale/static grasp"
                    )
                if warm_start:
                    runner.load(str(args.checkpoint), load_cfg={"actor": True})
                else:
                    runner.load(str(args.checkpoint))
            if args.mode == "train":
                if warm_start:
                    runner.save(str(args.output / "model_initial.pt"))
                runner.learn(
                    num_learning_iterations=args.iterations, init_at_random_ep_len=False
                )
            else:
                if args.checkpoint is None:
                    parser.error("Evaluation/view requires --checkpoint")
                policy = runner.get_inference_policy(device="cuda:0")
                hold_filter = None
                if args.hold_target_rate is not None:
                    from mjlab_grasp.safety_filter import HoldTargetFilter

                    hold_filter = HoldTargetFilter(env, term, args.hold_target_rate)
                if args.mode == "view":
                    viewed_policy = (
                        (lambda obs: hold_filter(policy(obs)))
                        if hold_filter is not None else policy
                    )
                    GraspPolicyViewer(wrapper, viewed_policy, duration=args.view_seconds).run()
                else:
                    obs = wrapper.get_observations()
                    successes = 0
                    stable_run = torch.zeros(env.num_envs, device=env.device)
                    completed = successful_episodes = dropped_episodes = 0
                    required_hold = 1.0 if version in ("v2", "v3", "v4", "v5") else 0.5
                    max_stable_hold = 0.0
                    over_speed_env_steps = 0
                    rows = []
                    for step in range(args.steps):
                        with torch.inference_mode():
                            actions = policy(obs)
                            applied_actions = (
                                hold_filter(actions) if hold_filter is not None else actions
                            )
                        obs, reward, done, _info = wrapper.step(applied_actions)
                        done = done.bool()
                        # Evaluate true terminal states before explicitly resetting.
                        stable = stable_success(env).bool()
                        speed_ok = joint_speed_ok(env).bool()
                        if version in ("v3", "v4", "v5"):
                            stable &= speed_ok
                        over_speed_env_steps += int((~speed_ok).sum())
                        stable_run = torch.where(
                            stable,
                            stable_run + env.step_dt,
                            torch.zeros_like(stable_run),
                        )
                        completed += int(done.sum())
                        failures = env.termination_manager.terminated
                        dropped_episodes += int((done & failures).sum())
                        successful_episodes += int(
                            (
                                done & ~failures & (stable_run >= required_hold - 1e-6)
                            ).sum()
                        )
                        max_stable_hold = max(max_stable_hold, float(stable_run.max()))
                        successes += (
                            int(stable.sum())
                            if version in ("v3", "v4", "v5")
                            else int(strict_success(env).sum())
                        )
                        obj = env.scene["object"]
                        rows.append(
                            {
                                "step": step,
                                "episode_time_s": float(
                                    env.episode_length_buf[0] * env.step_dt
                                ),
                                "position_error_mm": float(
                                    relative_error(env)[0].norm() * 1000
                                ),
                                "rotation_error_deg": float(
                                    torch.rad2deg(rotation_error(env)[0])
                                ),
                                "height_m": float(
                                    obj.data.root_link_pos_w[0, 2]
                                    - env.scene.env_origins[0, 2]
                                ),
                                "linear_speed_m_s": float(
                                    obj.data.root_link_lin_vel_w[0].norm()
                                ),
                                "angular_speed_rad_s": float(
                                    obj.data.root_link_ang_vel_w[0].norm()
                                ),
                                "raw_action_abs_max": float(actions[0].abs().max()),
                                "max_finger_speed_rad_s": float(
                                    env.scene["hand"].data.joint_vel[0].abs().max()
                                ),
                                "target_command_speed_rad_s": float(
                                    term._target_step[0].abs().max() / env.step_dt
                                ) if version == "v5" else (
                                    float(hold_filter.last_step[0].abs().max() / env.step_dt)
                                    if hold_filter is not None else None
                                ),
                                "raw_action_over_one_fraction": float(
                                    (actions[0].abs() > 1).float().mean()
                                ),
                                "stable_hold_s": float(stable_run[0]),
                                "reset": bool(done[0]),
                            }
                        )
                        if done.any():
                            reset_ids = done.nonzero(as_tuple=False).flatten()
                            env.reset(env_ids=reset_ids)
                            if hold_filter is not None:
                                hold_filter.reset(reset_ids)
                            stable_run[done] = 0
                            obs = wrapper.get_observations()
                    with (args.output / "trajectory_env0.csv").open(
                        "w", newline=""
                    ) as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                    (args.output / "evaluation.json").write_text(
                        json.dumps(
                            {
                                "task_version": version,
                                "hold_target_rate_rad_s": args.hold_target_rate,
                                "actuator_damping_Nm_s_rad": args.actuator_damping,
                                "steps": args.steps,
                                "num_envs": env.num_envs,
                                "completed_episodes": completed,
                                "successful_episodes": successful_episodes,
                                "dropped_episodes": dropped_episodes,
                                "episode_success_rate": successful_episodes / completed
                                if completed
                                else None,
                                "longest_stable_hold_s": max_stable_hold,
                                "required_stable_hold_s": required_hold,
                                "joint_speed_limit_rad_s": JOINT_SPEED_LIMIT
                                if version in ("v3", "v4", "v5")
                                else None,
                                "over_speed_env_steps": over_speed_env_steps,
                                "successful_env_steps": successes,
                                "success_criteria": "After lift: height > 8 cm, position error < 5 mm, rotation < 15 degrees, linear speed < 0.02 m/s, angular speed < 0.3 rad/s continuously for required_stable_hold_s immediately before timeout; dropped episodes fail"
                                + (
                                    "; every finger joint speed <= 0.2 rad/s throughout the hold window"
                                    if version in ("v3", "v4", "v5")
                                    else ""
                                ),
                                "note": "Deterministic reference-state evaluation; no perturbation robustness claim.",
                            },
                            indent=2,
                        )
                        + "\n"
                    )
    finally:
        env.close()
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
