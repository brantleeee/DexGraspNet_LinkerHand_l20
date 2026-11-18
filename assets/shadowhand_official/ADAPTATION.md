# Official DexGraspNet ShadowHand baseline

Source: https://github.com/PKU-EPIC/DexGraspNet/tree/bd1b13d7248729af117e1d46aaa6266b147a3c7b

`SOURCE.json` records the commit and SHA256 of each unmodified upstream file.
The upstream README describes dataset licensing (CC BY-NC 4.0); the upstream
OpenAI asset license is retained at `grasp_generation/open_ai_assets/LICENSE.md`.
The five bundled datasets contain 1,141 original ShadowHand grasp poses.
Object geometry is also from this exact upstream revision.

`mjlab_grasp/shadowhand.py` adapts the XML in memory for MuJoCo 3:

- Remove obsolete `apirate` and `nstack` attributes.
- Remove two wrist actuators and two wrist sensors referring to missing WRJ0/1
  joints. The original file contains a fixed palm and 22 finger joints.
- Resolve mesh paths and add an ideal arm wrapper and free object.
- Enable hand/object collisions, with self collision disabled. Use the existing
  validation protocol's zero collision margin, object density 500 kg/m³, solver,
  friction, and kinematic arm trajectory; those are experimental settings.

All 18 native finger actuators, their gains and force limits, inertias, finger
collision primitives, and four limited tendons are preserved. Source qpos is
neither clipped nor reprojected. This baseline tests transfer into MuJoCo, not
reproduction of the official Isaac Gym success rates. Local refinement and
generation of new ShadowHand grasps are outside this entry point.

The default `--physics-profile source` preserves the effective zero joint
damping/armature in the generation XML. Although `robot0:asset_class` declares
damping 0.1, armature 0.001 and frictionloss 0.001, the palm lacks a `childclass`
and those joint defaults are inactive. Use `--physics-profile asset-defaults`
for an explicit comparison that activates this upstream class on the palm
subtree. This is a changed physics profile, not an exact source reproduction;
both profiles retain all four tendon constraints and the original candidate.

Run a five-object, seven-case smoke test from the repository root:

```bash
/home/lee/robot_learning/mjlab/.venv/bin/python scripts/test_shadowhand_mujoco.py \
  --backend cpu --scale 0.06 --limit 2 \
  --output data/experiments/shadowhand_cpu_manual
```

Use `--backend mjlab` to explicitly select CUDA MuJoCo-Warp. Failures on that
backend do not silently fall back to CPU. Omit `--limit` and `--scale` to test
all 1,141 examples at their saved scales (7,987 independent trials).
Output directories must be new; interrupted runs are retained for diagnosis
and this standalone runner does not support resume.

For a two-object comparison, `--objects core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03
--jenga-mode transfer` uses the pinned mug ShadowHand grasp unchanged with the
local `2_of_Jenga_Classic_Game` mesh. Jenga is a cross-object transfer test:
the pinned five-object subset has no native ShadowHand Jenga grasp, and the
repository's Jenga `.npy` contains L20 joints. This result must not be labeled
an official Jenga grasp. `--jenga-mode native --jenga-result-file PATH` accepts
a separately supplied native ShadowHand Jenga pose file and rejects L20 poses.
The Jenga collision mesh comes from the local reconstructed visualization,
not from the pinned upstream five-object subset; its path and SHA256 are saved.

Each candidate gets source pose JSON, compiled runtime XML, state/contact NPZ
trajectories and case results. Root `summary.json` stays incomplete if a runtime
error occurs. `REPORT.md` is written only after the selected run completes.
