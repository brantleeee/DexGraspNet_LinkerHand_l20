"""Native MuJoCo counterpart of the pinned DexGraspNet Gym contact protocol.

Uses the exact Gym hand XML, exported runtime mechanics, and prepared raw/adjusted
joint targets. Contact shell semantics and solver integration are engine-specific.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from queue import SimpleQueue
import sys
import time
import threading
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / 'external/DexGraspNet_official/grasp_generation/open_ai_assets/hand/shadow_hand.xml'
INPUTS = ROOT / 'data/experiments/shadow_cup_jenga_isaacgym_inputs_20261102'
MECHANICS = ROOT / 'assets/shadowhand_official/ISAAC_GYM_RUNTIME_MECHANICS.json'
OBJECTS = {'mug': 'core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03', 'jenga': '2_of_Jenga_Classic_Game'}
LABELS = ('gy-', 'gy+', 'gx-', 'gx+', 'gz+', 'gz-')
ROTATIONS = [Rotation.identity(), Rotation.from_rotvec([0, 0, math.pi]),
             Rotation.from_rotvec([0, 0, math.pi / 2]), Rotation.from_rotvec([0, 0, -math.pi / 2]),
             Rotation.from_rotvec([math.pi / 2, 0, 0]), Rotation.from_rotvec([-math.pi / 2, 0, 0])]


def vector(values):
    return ' '.join(f'{float(x):.12g}' for x in values)


def expand(path):
    root = ET.parse(path).getroot()
    def include(parent):
        for child in list(parent):
            if child.tag == 'include':
                replacement = expand(path.parent / child.get('file'))
                offset = list(parent).index(child)
                parent.remove(child)
                for i, element in enumerate(replacement):
                    parent.insert(offset + i, element)
            else:
                include(child)
    include(root)
    return root


def inertial(body, mass, center, inertia):
    old = body.find('inertial')
    if old is not None:
        body.remove(old)
    tensor = np.asarray(inertia)
    ET.SubElement(body, 'inertial', mass=str(mass), pos=vector(center),
                  fullinertia=vector([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                     tensor[0, 1], tensor[0, 2], tensor[1, 2]]))


def build(obj, inputs, mechanics, substeps, contact_shell, drive_integration, shell_mode):
    scene = expand(HAND)
    integration_shell = contact_shell if shell_mode == 'continuous' else 0
    compiler = scene.find('compiler')
    mesh_dir = (HAND.parent / compiler.get('meshdir')).resolve()
    texture_dir = (HAND.parent / compiler.get('texturedir')).resolve()
    compiler.attrib.pop('meshdir', None)
    compiler.attrib.pop('texturedir', None)
    for mesh in scene.findall('asset/mesh'):
        mesh.set('file', str(mesh_dir / mesh.get('file')))
    for texture in scene.findall('asset/texture'):
        if texture.get('file'):
            texture.set('file', str(texture_dir / texture.get('file')))
    option = scene.find('option')
    option.attrib.pop('apirate', None)
    option.set('timestep', str(1 / (60 * substeps)))
    option.set('gravity', '0 -9.8 0')
    option.set('integrator', 'implicitfast')
    option.set('iterations', '50')
    scene.find('size').attrib.pop('nstack', None)
    for tag in ('contact', 'sensor', 'actuator'):
        element = scene.find(tag)
        if element is not None:
            scene.remove(element)
    actuators = ET.SubElement(scene, 'actuator')
    world = scene.find('worldbody')
    base = copy.deepcopy(world.find('body'))
    for body in list(world.findall('body')):
        world.remove(body)
    hand_props = mechanics['actual_hand']
    dofs = {p['name']: p for p in hand_props['dofs']}
    body_props = {p['name']: p for p in hand_props['bodies']}
    for body in base.iter('body'):
        p = body_props[body.get('name')]
        inertial(body, p['mass'], p['com'], p['inertia'])
    for joint in base.iter('joint'):
        p = dofs[joint.get('name')]
        joint.set('limited', 'true' if p['hasLimits'] else 'false')
        if p['hasLimits']:
            joint.set('range', vector([p['lower'], p['upper']]))
        joint.set('damping', '0')
        joint.set('armature', str(p['armature']))
        joint.set('frictionloss', str(p['friction']))
    native_tendons = scene.find('tendon')
    tendon_template = copy.deepcopy(native_tendons)
    scene.remove(native_tendons)
    tendons = ET.SubElement(scene, 'tendon')
    tendon_props = {p['name']: p for p in hand_props['tendons']}

    # URDF pieces have identity fixed-link frames; fuse their measured inertias.
    pieces = mechanics['actual_object']['bodies']
    mass = sum(p['mass'] for p in pieces)
    center = sum(p['mass'] * np.asarray(p['com']) for p in pieces) / mass
    inertia_sum = np.zeros((3, 3))
    for piece in pieces:
        delta = np.asarray(piece['com']) - center
        inertia_sum += np.asarray(piece['inertia']) + piece['mass'] * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    assets = scene.find('asset')
    meshes = sorted((inputs / 'assets' / obj).glob('convex_*.obj'))
    for i, mesh in enumerate(meshes):
        ET.SubElement(assets, 'mesh', name=f'object_part_{i}', file=str(mesh.resolve()))
    for direction in range(6):
        prefix = f'd{direction}:'
        mount = copy.deepcopy(base)
        mount.set('name', prefix + 'mount')
        for element in mount.iter():
            if element is mount:
                continue
            if element.get('name'):
                element.set('name', prefix + element.get('name'))
        for geom in mount.iter('geom'):
            visual = geom.get('class') == 'robot0:D_Vizual'
            geom.set('contype', '0' if visual else str(1 << (2 * direction)))
            geom.set('conaffinity', '0' if visual else str(1 << (2 * direction + 1)))
            geom.set('group', '1' if visual else '3')
            geom.set('rgba', '0.2 0.6 0.95 1')
            geom.set('friction', '3 0 0')
            geom.set('margin', str(integration_shell))
            # Detection shell, with force onset at zero geometric separation.
            geom.set('gap', str(integration_shell))
            geom.set('condim', '3')
        world.append(mount)
        for name, p in dofs.items():
            effort = p['effort']
            # With implicitfast, D=h*K gives backward-Euler integration of a
            # zero-damping spring in the unconstrained linear case:
            # (M+h^2*K)*v_next = M*v + h*K*(target-q).
            # This is a numerical surrogate for PhysX's implicit drive, not an
            # exact cross-engine equivalence in contact or force saturation.
            kv = p['damping'] + (p['stiffness'] / (60 * substeps) if drive_integration == 'implicit' else 0)
            attributes = dict(name=prefix + name, joint=prefix + name, kp=str(p['stiffness']), kv=str(kv),
                              ctrllimited='false', forcelimited='true' if effort < 1e20 else 'false')
            if effort < 1e20:
                attributes['forcerange'] = vector([-effort, effort])
            ET.SubElement(actuators, 'position', **attributes)
        for tendon in tendon_template:
            element = copy.deepcopy(tendon)
            p = tendon_props[element.get('name')]
            element.set('name', prefix + element.get('name'))
            # Gym uses a weak spring outside tendon limits, not a hard constraint.
            element.set('limited', 'false')
            element.set('stiffness', str(p['limit_stiffness']))
            element.set('damping', str(p['damping']))
            element.set('springlength', vector([p['fixed_lower_limit'], p['fixed_upper_limit']]))
            for joint in element.findall('joint'):
                joint.set('joint', prefix + joint.get('joint'))
            tendons.append(element)
        body = ET.SubElement(world, 'body', name=prefix + 'object')
        ET.SubElement(body, 'freejoint', name=prefix + 'object_free')
        inertial(body, mass, center, inertia_sum)
        for i in range(len(meshes)):
            ET.SubElement(body, 'geom', name=prefix + f'object_geom_{i}', type='mesh', mesh=f'object_part_{i}',
                          contype=str(1 << (2 * direction + 1)), conaffinity=str(1 << (2 * direction)),
                          condim='3', margin=str(integration_shell), gap=str(integration_shell), friction='3 0 0',
                          group='4', rgba='0.95 0.65 0.15 1')
    return scene


def initialize(scene, inputs, index, variant):
    scene = copy.deepcopy(scene)
    scale = float(inputs['scales'][index])
    for mesh in scene.findall('asset/mesh'):
        if mesh.get('name', '').startswith('object_part_'):
            mesh.set('scale', vector([scale] * 3))
    quat = inputs['rotations'][index]
    hand_rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
    for i, rotation in enumerate(ROTATIONS):
        offset = np.array([i % 3 * 1.0, 0, i // 3 * 1.0])
        mount = scene.find(f"worldbody/body[@name='d{i}:mount']")
        mount.set('pos', vector(offset + rotation.apply(inputs['translations'][index])))
        mount.attrib.pop('euler', None)
        mount.set('quat', vector((rotation * hand_rotation).as_quat()[[3, 0, 1, 2]]))
        obj = scene.find(f"worldbody/body[@name='d{i}:object']")
        obj.set('pos', vector(offset))
        obj.set('quat', vector(rotation.as_quat()[[3, 0, 1, 2]]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
    data = mujoco.MjData(model)
    # Prepared qpos ordering is identical to the official validator's joint list.
    names = ['FFJ3', 'FFJ2', 'FFJ1', 'FFJ0', 'MFJ3', 'MFJ2', 'MFJ1', 'MFJ0',
             'RFJ3', 'RFJ2', 'RFJ1', 'RFJ0', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'LFJ0',
             'THJ4', 'THJ3', 'THJ2', 'THJ1', 'THJ0']
    angles = inputs['force_qpos' if variant == 'official' else 'raw_qpos'][index]
    for i in range(6):
        for name, angle in zip(names, angles):
            joint = model.joint(f'd{i}:robot0:{name}')
            data.qpos[joint.qposadr] = angle
            data.ctrl[model.actuator(f'd{i}:robot0:{name}').id] = angle
    mujoco.mj_forward(model, data)
    return model, data


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=INPUTS)
    parser.add_argument('--mechanics', type=Path, default=MECHANICS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objects', nargs='+', choices=OBJECTS, default=list(OBJECTS))
    parser.add_argument('--variants', nargs='+', choices=('official', 'no-force'), default=['official'])
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--sim-steps', type=int, default=1000)
    parser.add_argument('--substeps', type=int, default=512, help='Numerical integration substeps per 1/60 s control step')
    parser.add_argument('--contact-shell', type=float, default=0.01)
    parser.add_argument('--drive-integration', choices=('implicit', 'native'), default='native',
                        help='implicit uses timestep compensation for stiff position drives; native uses kv=0')
    parser.add_argument('--contact-shell-mode', choices=('final', 'continuous'), default='final',
                        help='final omits force-free proximity contacts during stepping and queries the shell at the end')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--autostart', action='store_true')
    parser.add_argument('--display-speed', type=float, default=1)
    parser.add_argument('--viewer-seconds', type=float, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if args.sim_steps < 1 or args.substeps < 1 or not np.isfinite(args.contact_shell) or args.contact_shell < 0:
        parser.error('positive steps/substeps and nonnegative finite contact shell required')
    if args.gui and (len(args.objects) != 1 or len(args.variants) != 1 or len(args.indices or []) != 1):
        parser.error('GUI requires one object, variant and candidate index')
    if args.display_speed <= 0 or not np.isfinite(args.display_speed) or args.viewer_seconds < 0:
        parser.error('invalid viewer speed/duration')
    args.output.mkdir(parents=True)
    runtime = json.loads(args.mechanics.read_text())
    summary = {'completed': False, 'engine': 'native MuJoCo', 'version': mujoco.__version__,
               'protocol': {'control_steps': args.sim_steps, 'control_dt': 1 / 60,
                            'simulated_seconds': args.sim_steps / 60, 'integration_substeps': args.substeps,
                            'drive_integration': args.drive_integration,
                            'contact_shell_mode': args.contact_shell_mode,
                            'drive_kp': 1000, 'physical_drive_damping': 0,
                            'numerical_actuator_kv': 1000 / (60 * args.substeps) if args.drive_integration == 'implicit' else 0,
                            'contact_shell': args.contact_shell, 'success': 'six final contact lists and saved E_pen < 0.001'},
               'mechanics_source': str(args.mechanics.resolve()),
               'mechanics_sha256': hashlib.sha256(args.mechanics.read_bytes()).hexdigest(),
               'gym_hand_xml_sha256': hashlib.sha256(HAND.read_bytes()).hexdigest(),
               'input_sha256': {short: hashlib.sha256((args.inputs / (OBJECTS[short] + '.npz')).read_bytes()).hexdigest()
                                for short in args.objects},
               'limitations': ['MuJoCo contact margin/gap approximates PhysX contact-offset shell; solver semantics differ',
                               'URDF fixed convex links fused with measured aggregate mass and inertia',
                               'PhysX joint velocity caps and contact prediction are not reproduced exactly',
                               ('Implicit surrogate adds kv=h*kp; not an exact contact/force-saturation match' if args.drive_integration == 'implicit'
                                else 'Native zero-damping drive requires more integration substeps than PhysX')], 'runs': []}
    save(args.output / 'summary.json', summary)
    for short in args.objects:
        obj = OBJECTS[short]
        mechanics = next(r['mechanism'] for r in runtime['runs'] if r['object'] == obj)
        with np.load(args.inputs / (obj + '.npz')) as prepared:
            inputs = dict(prepared)
        indices = args.indices if args.indices is not None else list(range(len(inputs['scales'])))
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(inputs['scales']) for i in indices):
            parser.error('invalid/duplicate indices')
        template = build(obj, args.inputs.resolve(), mechanics, args.substeps, args.contact_shell, args.drive_integration, args.contact_shell_mode)
        for variant in args.variants:
            run = {'object': obj, 'variant': variant, 'candidates': []}
            summary['runs'].append(run)
            for index in indices:
                folder = args.output / short / variant / f'candidate_{index:04d}'
                folder.mkdir(parents=True)
                model, data = initialize(template, inputs, index, variant)
                mujoco.mj_saveLastXML(str(folder / 'scene.xml'), model)
                print(f'{short} {variant} #{index}: {args.sim_steps} steps x 6 directions; substeps={args.substeps}', flush=True)
                result = simulate(model, data, args, folder)
                result.update(index=index, E_pen=float(inputs['E_pen'][index]), scale=float(inputs['scales'][index]))
                result['estimated_pass'] = result['E_pen'] < 0.001
                result['simulated_pass'] = result['numerically_valid'] and all(c['final_contact'] for c in result['directions'])
                result['valid'] = result['simulated_pass'] and result['estimated_pass']
                run['candidates'].append(result)
                save(folder / 'results.json', result)
                save(args.output / 'summary.json', summary)
                print(f'valid={result["valid"]}; numerical={result["numerically_valid"]}; time={data.time:.6f}s', flush=True)
    summary['completed'] = True
    save(args.output / 'summary.json', summary)
    lines = ['# MuJoCo comparison using the DexGraspNet Gym protocol', '',
             f'{args.sim_steps} control steps, {args.sim_steps / 60:.6f} simulated seconds, {args.substeps} integration substeps.', '',
             '| Object | Variant | Candidates | Simulated pass | Valid | Numerical failure |', '|---|---|---:|---:|---:|---:|']
    for run in summary['runs']:
        cs = run['candidates']
        lines.append(f'| {run["object"]} | {run["variant"]} | {len(cs)} | {sum(c["simulated_pass"] for c in cs)} | '
                     f'{sum(c["valid"] for c in cs)} | {sum(not c["numerically_valid"] for c in cs)} |')
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f'Complete: {args.output / "REPORT.md"}', flush=True)


def simulate(model, data, args, folder):
    # Defined below so headless and GUI share the exact same stepping loop.
    object_ids = [model.body(f'd{i}:object').id for i in range(6)]
    initial = [data.xpos[i].copy() for i in object_ids]
    max_displacement = np.zeros(6)
    viewer, events = None, SimpleQueue()
    render_threads = []
    paused = args.gui and not args.autostart
    direction = 0
    if args.gui:
        import mujoco.viewer as mj_viewer
        import glfw
        existing_threads = set(threading.enumerate())
        viewer = mj_viewer.launch_passive(model, data, key_callback=events.put, show_left_ui=False, show_right_ui=False)
        render_threads = [t for t in threading.enumerate() if t not in existing_threads
                          and '_launch_internal' in t.name]
        viewer.cam.lookat[:] = initial[0]
        viewer.cam.distance = 0.35
        viewer.cam.azimuth, viewer.cam.elevation = -45, -20
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[3:5] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
        print('SPACE: pause/resume | 1..6: focus scene | MuJoCo mouse controls: rotate/pan/zoom', flush=True)
    history = {'time': [float(data.time)], 'qpos': [data.qpos.copy()],
               'qvel': [data.qvel.copy()], 'ctrl': [data.ctrl.copy()]}
    bad = False
    started = time.monotonic()
    steps = 0
    peak_joint_speed = 0.0
    driven_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    try:
        for step in range(args.sim_steps):
            if viewer:
                while True:
                    if not viewer.is_running():
                        raise RuntimeError('Viewer closed before completion')
                    while not events.empty():
                        key = events.get()
                        if key == glfw.KEY_SPACE:
                            paused = not paused
                        elif ord('1') <= key <= ord('6'):
                            direction = key - ord('1')
                            with viewer.lock():
                                viewer.cam.lookat[:] = initial[direction]
                    if not paused:
                        break
                    viewer.sync()
                    time.sleep(0.01)
            frame_start = time.monotonic()
            if viewer:
                with viewer.lock():
                    mujoco.mj_step(model, data, nstep=args.substeps)
            else:
                mujoco.mj_step(model, data, nstep=args.substeps)
            steps += 1
            mujoco.mj_forward(model, data)
            peak_joint_speed = max(peak_joint_speed, float(np.abs(data.qvel[driven_dofs]).max()))
            for i, body in enumerate(object_ids):
                max_displacement[i] = max(max_displacement[i], np.linalg.norm(data.xpos[body] - initial[i]))
            for key in history:
                history[key].append(float(data.time) if key == 'time' else getattr(data, key).copy())
            bad = (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or
                   any(data.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_BADQPOS,
                       mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)) or
                   abs(data.time - steps / 60) > 1e-5)
            if bad:
                break
            if steps % 100 == 0:
                save(folder / 'progress.json', {'steps': steps, 'requested_steps': args.sim_steps, 'time': float(data.time)})
                print(f'Progress: {steps}/{args.sim_steps}; engine time={data.time:.6f}s', flush=True)
            if viewer:
                viewer.sync()
                time.sleep(max(0, 1 / (60 * args.display_speed) - (time.monotonic() - frame_start)))
        if args.contact_shell_mode == 'final':
            model.geom_margin[:] = args.contact_shell
            model.geom_gap[:] = args.contact_shell
            mujoco.mj_forward(model, data)
        contacts = [[] for _ in range(6)]
        active = [0] * 6
        for c in data.contact:
            body1, body2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            for i, oid in enumerate(object_ids):
                if body1 == oid or body2 == oid:
                    contacts[i].append(float(c.dist))
                    active[i] += int(c.efc_address >= 0)
        result = {'executed_steps': steps, 'actual_simulated_seconds': float(data.time),
                  'numerically_valid': not bad and steps == args.sim_steps, 'wall_seconds': time.monotonic() - started,
                  'peak_joint_speed_rad_s': peak_joint_speed,
                  'directions': [{'rotation_index': i, 'gravity_label': LABELS[i], 'final_contact': bool(contacts[i]),
                                  'final_contact_count': len(contacts[i]), 'force_constraint_contacts': active[i],
                                  'final_contact_distances_m': contacts[i], 'max_displacement_m': max_displacement[i],
                                  'object_mass_kg': float(model.body_mass[oid])} for i, oid in enumerate(object_ids)]}
        np.savez_compressed(folder / 'trajectory.npz', **{k: np.asarray(v) for k, v in history.items()})
        save(folder / 'simulation.json', result)
        if viewer:
            print('Simulation saved. Close window to finish; 1..6 selects a scene.', flush=True)
            hold = time.monotonic()
            while viewer.is_running() and (not args.viewer_seconds or time.monotonic() - hold < args.viewer_seconds):
                while not events.empty():
                    key = events.get()
                    if ord('1') <= key <= ord('6'):
                        viewer.cam.lookat[:] = initial[key - ord('1')]
                viewer.sync()
                time.sleep(0.01)
        return result
    finally:
        if viewer:
            viewer.close()
            for thread in render_threads:
                thread.join(timeout=10)


if __name__ == '__main__':
    main()
