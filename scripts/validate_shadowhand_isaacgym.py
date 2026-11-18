"""Run the pinned official DexGraspNet IsaacValidator on prepared native grasps.

Launch with Python 3.8 and the NVIDIA Isaac Gym Preview 4 PYTHONPATH.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/DexGraspNet_official/grasp_generation"
sys.path.insert(0, str(SOURCE))
from utils.isaac_validator import IsaacValidator, gym
from isaacgym import gymapi
import numpy as np

OBJECTS = {'mug': 'core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03',
           'jenga': '2_of_Jenga_Classic_Game'}


def contact_flags(sim):
    counts = []
    for i, env in enumerate(sim.envs):
        hand, obj = sim.hand_rigid_body_sets[i], sim.obj_rigid_body_sets[i]
        counts.append(sum((c[2] in hand and c[3] in obj) or
                          (c[3] in hand and c[2] in obj)
                          for c in gym.get_env_rigid_contacts(env)))
    return counts


def actor_mechanics(env, actor):
    dofs = gym.get_actor_dof_properties(env, actor)
    tendons = gym.get_actor_tendon_properties(env, actor)
    bodies = gym.get_actor_rigid_body_properties(env, actor)
    return {
        'dofs': [dict(name=gym.get_actor_dof_names(env, actor)[i],
                      **{key: dofs[key][i].item() for key in dofs.dtype.names}) for i in range(len(dofs))],
        'tendons': [dict(name=gym.get_actor_tendon_name(env, actor, i),
                         **{key: getattr(p, key) for key in dir(p) if not key.startswith('_')
                            and isinstance(getattr(p, key), (float, int, bool))}) for i, p in enumerate(tendons)],
        'bodies': [{'name': gym.get_actor_rigid_body_names(env, actor)[i], 'mass': p.mass,
                    'com': [p.com.x, p.com.y, p.com.z],
                    'inertia': [[p.inertia.x.x, p.inertia.x.y, p.inertia.x.z],
                                [p.inertia.y.x, p.inertia.y.y, p.inertia.y.z],
                                [p.inertia.z.x, p.inertia.z.y, p.inertia.z.z]]}
                   for i, p in enumerate(bodies)]}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def report(summary, output):
    protocol = summary['protocol']
    lines = ['# DexGraspNet Isaac Gym validation', '',
             'NVIDIA Isaac Gym Preview 4; GPU PhysX; pinned official assets and contact criterion.',
             f"Steps per direction: {protocol['sim_steps']}; simulated duration: {protocol['sim_seconds']:.6f} s; "
             f"duration relative to official 100-step test: {protocol['duration_multiplier']:g}x.",
             'Dynamic success: final hand-object contact in all six directions. Valid additionally requires saved E_pen < 0.001.', '',
             '| Object | Variant | Candidates | Dynamic pass | Valid pass | Actual engine duration (s) |',
             '|---|---|---:|---:|---:|---:|']
    for run in summary['runs']:
        lines.append(f'| {run["object"]} | {run["variant"]} | {len(run["candidates"])} | '
                     f'{run["simulated_pass"]} | {run["valid_pass"]} | {run["timing"]["simulated_seconds"]:.6f} |')
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/experiments/shadow_cup_jenga_isaacgym_inputs_20261102')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objects', choices=OBJECTS, nargs='+', default=list(OBJECTS))
    parser.add_argument('--variants', choices=('official', 'no-force'), nargs='+', default=['official', 'no-force'])
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--sim-steps', type=int, default=1000, help='Physics steps per direction (default: 1000; original protocol: 100)')
    parser.add_argument('--autostart', action='store_true', help='Start GUI simulation without pressing SPACE')
    parser.add_argument('--viewer-seconds', type=float, default=0,
                        help='Close GUI this many seconds after completion; default keep open')
    parser.add_argument('--display-speed', type=float, default=1,
                        help='GUI simulated-time/wall-time playback speed (default: 1)')
    args = parser.parse_args()
    if args.sim_steps <= 0:
        parser.error('--sim-steps must be positive')
    if not np.isfinite(args.display_speed) or args.display_speed <= 0:
        parser.error('--display-speed must be positive and finite')
    if not np.isfinite(args.viewer_seconds) or args.viewer_seconds < 0:
        parser.error('--viewer-seconds must be finite and nonnegative')
    inputs, output = args.inputs.resolve(), args.output.resolve()
    if output.exists():
        parser.error('output already exists; choose a new directory')
    if args.gui and (len(args.objects) != 1 or len(args.variants) != 1 or len(args.indices or []) != 1):
        parser.error('--gui requires one object, one variant and one index')
    output.mkdir(parents=True)
    summary = {'completed': False, 'engine': 'Isaac Gym Preview 4 / GPU PhysX',
               'source_commit': 'bd1b13d7248729af117e1d46aaa6266b147a3c7b',
               'source_validator_sha256': hashlib.sha256((SOURCE / 'utils/isaac_validator.py').read_bytes()).hexdigest(),
               'input_manifest': json.loads((inputs / 'manifest.json').read_text()),
               'protocol': {'dt': 1 / 60, 'substeps': 2, 'sim_steps': args.sim_steps,
                            'sim_seconds': args.sim_steps / 60, 'stiffness': 1000,
                            'original_protocol_steps': 100, 'duration_multiplier': args.sim_steps / 100,
                            'damping': 0, 'friction': 3, 'density': 500, 'contact_offset': 0.01,
                            'gravity': [0, -9.8, 0], 'penetration_energy_threshold': 0.001,
                            'success': 'final hand-object contact in all six rotated scenes and saved E_pen < 0.001'},
               'runs': []}
    save(output / 'summary.json', summary)
    for short_name in args.objects:
        obj = OBJECTS[short_name]
        with np.load(inputs / f'{obj}.npz') as arrays:
            data = dict(arrays)
        indices = args.indices if args.indices is not None else list(range(len(data['scales'])))
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(data['scales']) for i in indices):
            parser.error('invalid or duplicate candidate indices')
        for variant in args.variants:
            print(f'Running {short_name} {variant}: {len(indices)} candidates x 6 directions', flush=True)
            sim = IsaacValidator(gpu=0, sim_step=args.sim_steps, mode='gui' if args.gui else 'direct')
            if sim.sim is None:
                raise RuntimeError('Isaac Gym failed to create the GPU PhysX simulator')
            try:
                sim.set_asset(str(SOURCE / 'open_ai_assets'), 'hand/shadow_hand.xml',
                              str(inputs / 'assets' / obj), 'coacd.urdf')
                qpos = data['force_qpos'] if variant == 'official' else data['raw_qpos']
                for index in indices:
                    sim.add_env(data['rotations'][index], data['translations'][index],
                                qpos[index], float(data['scales'][index]))
                initial = [gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
                           for env, actor in zip(sim.envs, sim.obj_handles)]
                masses = [sum(float(p.mass) for p in gym.get_actor_rigid_body_properties(env, actor))
                          for env, actor in zip(sim.envs, sim.obj_handles)]
                display = None
                sim_time_start = gym.get_sim_time(sim.sim)
                wall_start = time.monotonic()
                if args.gui:
                    from shadowhand_gym_viewer import GymGraspViewer
                    display = GymGraspViewer(sim, gym, output, args.autostart, args.viewer_seconds, args.display_speed)
                    display.run()
                    upstream_flags = [count > 0 for count in contact_flags(sim)]
                else:
                    upstream_flags = sim.run_sim()
                # Wait for the final PhysX step before collecting diagnostic state.
                gym.fetch_results(sim.sim, True)
                sim_elapsed = gym.get_sim_time(sim.sim) - sim_time_start
                expected_elapsed = args.sim_steps * sim.sim_params.dt
                if abs(sim_elapsed - expected_elapsed) > 0.01:
                    raise RuntimeError(f'Engine advanced {sim_elapsed}s; expected {expected_elapsed}s')
                counts = contact_flags(sim)
                synchronized_flags = [count > 0 for count in counts]
                run = {'object': obj, 'variant': variant, 'candidates': [],
                       'execution': 'gui_synchronized_step_loop' if args.gui else 'upstream_run_sim',
                       'timing': {'executed_steps': display.steps if display is not None else args.sim_steps,
                                  'simulated_seconds': sim_elapsed, 'expected_seconds': expected_elapsed,
                                  'wall_seconds_including_pause': time.monotonic() - wall_start},
                       'mechanism': {'hand_dofs': gym.get_asset_dof_count(sim.hand_asset),
                                     'actual_hand': actor_mechanics(sim.envs[0], sim.hand_handles[0]),
                                     'actual_object': actor_mechanics(sim.envs[0], sim.obj_handles[0]),
                                     'hand_tendons': gym.get_asset_tendon_count(sim.hand_asset),
                                     'hand_rigid_shapes': gym.get_asset_rigid_shape_count(sim.hand_asset),
                                     'object_rigid_bodies': gym.get_asset_rigid_body_count(sim.obj_asset)},
                       'upstream_and_synchronized_flags_agree': upstream_flags == synchronized_flags if not args.gui else None}
                states = {}
                for position, index in enumerate(indices):
                    begin = position * 6
                    flags = upstream_flags[begin:begin + 6]
                    estimated = bool(data['E_pen'][index] < 0.001)
                    entry = {'index': index, 'scale': float(data['scales'][index]),
                             'E_pen': float(data['E_pen'][index]), 'estimated_pass': estimated,
                             'simulated_pass': bool(all(flags)), 'valid': bool(all(flags) and estimated),
                             'directions': []}
                    for direction in range(6):
                        k = begin + direction
                        final = gym.get_actor_rigid_body_states(sim.envs[k], sim.obj_handles[k], gymapi.STATE_ALL).copy()
                        xyz = np.column_stack([final['pose']['p'][n] for n in ('x', 'y', 'z')])
                        start = np.column_stack([initial[k]['pose']['p'][n] for n in ('x', 'y', 'z')])
                        states[f'candidate_{index:02d}_direction_{direction}_initial'] = initial[k]
                        states[f'candidate_{index:02d}_direction_{direction}_final'] = final
                        entry['directions'].append({'rotation_index': direction,
                            'final_contact': bool(flags[direction]), 'synchronized_final_contact': synchronized_flags[k],
                            'final_contact_count': int(counts[k]), 'object_mass_kg': masses[k],
                            'max_body_displacement_m': float(np.linalg.norm(xyz - start, axis=1).max()),
                            'finite_state': bool(np.isfinite(xyz).all())})
                    run['candidates'].append(entry)
                run['simulated_pass'] = sum(e['simulated_pass'] for e in run['candidates'])
                run['valid_pass'] = sum(e['valid'] for e in run['candidates'])
                summary['runs'].append(run)
                np.savez(output / f'{short_name}_{variant}_states.npz', **states)
                save(output / 'summary.json', summary)
                print(f'{short_name} {variant}: simulated={run["simulated_pass"]}/{len(indices)} valid={run["valid_pass"]}/{len(indices)}', flush=True)
                if display is not None:
                    summary['completed'] = True
                    save(output / 'summary.json', summary)
                    report(summary, output)
                    display.hold()
            finally:
                if sim.has_viewer:
                    gym.destroy_viewer(sim.viewer)
                    sim.has_viewer = False
                sim.destroy()
    summary['completed'] = True
    save(output / 'summary.json', summary)
    report(summary, output)
    print(f'Complete: {output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
