"""MuJoCo checks of URDF-feasible grasps, six gravity directions, and wrist lift.

No object weld, support plane, adhesion, or external object holding force is used.
Saved poses are projected onto the URDF limits and mimic equations before testing;
the source pose's feasibility is reported separately, never silently assumed.
"""

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'grasp_generation'))
from utils.adaptive_compliance import ComplianceConfig
from utils.mujoco_compliance import MujocoJointCompliance, independent_limits
from utils.grasp_results import verify_mesh_identity


def vector(values):
    return ' '.join(f'{float(x):.12g}' for x in values)


def project_pose(robot, saved):
    joints = {j.get('name'): j for j in robot.findall('joint') if j.get('type') != 'fixed'}
    target = {n: float(saved[n]) for n in joints}
    violations, mimic_errors = {}, {}
    for name, joint in joints.items():
        limit = joint.find('limit')
        lo, hi = float(limit.get('lower')), float(limit.get('upper'))
        violations[name] = max(lo-target[name], target[name]-hi, 0.)
        target[name] = float(np.clip(target[name], lo, hi))
    for name, joint in joints.items():
        mimic = joint.find('mimic')
        if mimic is None:
            continue
        parent = mimic.get('joint')
        ratio, offset = float(mimic.get('multiplier', '1')), float(mimic.get('offset', '0'))
        mimic_errors[name] = abs(saved[name] - (ratio*saved[parent] + offset))
        limit = joint.find('limit')
        lo, hi = float(limit.get('lower')), float(limit.get('upper'))
        parent_bounds = sorted(((lo-offset)/ratio, (hi-offset)/ratio))
        target[parent] = float(np.clip(target[parent], *parent_bounds))
        target[name] = ratio*target[parent]+offset
    return target, {
        'source_pose_feasible': max([*violations.values(), *mimic_errors.values()]) < 1e-4,
        'joint_limit_violation_rad': {n: v for n, v in violations.items() if v > 1e-6},
        'mimic_equation_error_rad': mimic_errors,
        'projected_joint_targets': target,
        'maximum_joint_correction_rad': max(abs(target[n]-saved[n]) for n in target),
    }


def build_template(root, output, object_code, self_collision=False, mesh_sdf=False):
    urdf_path = root/'grasp_generation/mjcf/linkerhand_l20_right.urdf'
    robot = ET.parse(urdf_path).getroot()
    imported = copy.deepcopy(robot)
    compiler = ET.SubElement(ET.SubElement(imported, 'mujoco'), 'compiler')
    compiler.set('discardvisual', 'false')
    compiler.set('fusestatic', 'false')
    for mesh in imported.findall('.//mesh'):
        mesh.set('filename', str(urdf_path.parent/mesh.get('filename')))
    hand = mujoco.MjModel.from_xml_string(ET.tostring(imported, encoding='unicode'))
    imported_path = output/'hand_import.xml'
    mujoco.mj_saveLastXML(str(imported_path), hand)
    scene = ET.parse(imported_path).getroot()
    scene.set('model', 'l20_grasp_dynamics')
    ET.SubElement(scene, 'option', timestep='0.001', integrator='implicitfast',
                  solver='Newton', iterations='100', tolerance='1e-10', gravity='0 0 0')
    world = scene.find('worldbody')
    base = world.find("body[@name='base_link']")
    world.remove(base)
    wrist = ET.SubElement(world, 'body', name='wrist', mocap='true')
    wrist.append(base)
    for geom in base.iter('geom'):
        visual = geom.get('contype') == '0'
        geom.set('rgba', '0.45 0.7 0.85 1')
        if not visual:
            geom.set('contype', '1')
            geom.set('conaffinity', '3' if self_collision else '2')
            geom.set('group', '3')
            geom.set('condim', '3')
            geom.set('solref', '0.005 1')
            geom.set('solimp', '0.95 0.99 0.001')
            geom.set('margin', '0')
    for joint in base.iter('joint'):
        joint.set('damping', '0.02')
    actuators = ET.SubElement(scene, 'actuator')
    for joint in robot.findall('joint'):
        if joint.get('type') == 'fixed' or joint.find('mimic') is not None:
            continue
        lim = joint.find('limit')
        effort = float(lim.get('effort'))
        ET.SubElement(actuators, 'general', name='drive_'+joint.get('name'),
                      joint=joint.get('name'), gainprm='5', biastype='affine',
                      biasprm='0 -5 -0.05', ctrllimited='true',
                      ctrlrange=vector(independent_limits(robot, joint.get('name'))),
                      forcelimited='true', forcerange=vector([-effort, effort]))
    mesh_path = root/'data/meshdata'/object_code/'coacd/decomposed.obj'
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    pieces = [mesh] if mesh_sdf else mesh.split(only_watertight=False)
    asset_dir = output/'collision_parts'
    asset_dir.mkdir(exist_ok=True)
    asset = scene.find('asset')
    obj = ET.SubElement(world, 'body', name='object')
    ET.SubElement(obj, 'freejoint', name='object_free')
    for i, piece in enumerate(pieces):
        hull = piece if mesh_sdf else piece.convex_hull
        file = asset_dir/f'part_{i:03d}.obj'
        hull.export(file)
        ET.SubElement(asset, 'mesh', name=f'object_part_{i}', file=str(file.resolve()))
        ET.SubElement(obj, 'geom', name=f'object_geom_{i}', type='sdf' if mesh_sdf else 'mesh',
                      mesh=f'object_part_{i}', density='500', contype='2',
                      conaffinity='1', condim='3', solref='0.005 1',
                      solimp='0.95 0.99 0.001', margin='0', group='4', rgba='0.35 0.75 0.25 1')
    ET.SubElement(asset, 'mesh', name='object_visual', file=str(mesh_path.resolve()))
    ET.SubElement(obj, 'geom', name='object_visual_geom', type='mesh', mesh='object_visual',
                  contype='0', conaffinity='0', density='0', group='2', rgba='0.35 0.75 0.25 0.3')
    ET.SubElement(world, 'light', pos='0 0 2', dir='0 0 -1')
    if mesh_sdf:
        for geom in base.iter('geom'):
            if geom.get('mesh') and (int(geom.get('contype','1')) or int(geom.get('conaffinity','1'))):
                geom.set('type','sdf')
        equality=scene.find('equality')
        if equality is None:equality=ET.SubElement(scene,'equality')
        for joint in robot.findall('joint'):
            mimic=joint.find('mimic')
            if mimic is not None:
                existing=next((e for e in equality.findall('joint') if e.get('joint1')==joint.get('name')),None)
                if existing is not None:
                    if existing.get('joint2')!=mimic.get('joint'):raise ValueError('Imported mimic parent mismatch')
                    continue
                ET.SubElement(equality,'joint',name='physical_mimic_'+joint.get('name'),joint1=joint.get('name'),
                    joint2=mimic.get('joint'),polycoef=vector([float(mimic.get('offset','0')),float(mimic.get('multiplier','1')),0,0,0]),solref='0.005 1')
    return scene, robot, len(pieces)


def run_trial(scene, robot, grasp, index, friction, case, output, seconds,
              controller_mode='fixed', compliance_config=None, display=None):
    import transforms3d

    xml = copy.deepcopy(scene)
    q = grasp['qpos']
    projected, feasibility = project_pose(robot, q)
    position = np.array([q['WRJT'+a] for a in 'xyz'])
    quat = transforms3d.euler.euler2quat(*[q['WRJR'+a] for a in 'xyz'])
    wrist = xml.find("worldbody/body[@name='wrist']")
    wrist.set('pos', vector(position))
    wrist.set('quat', vector(quat))
    for mesh in xml.findall("asset/mesh"):
        if mesh.get('name').startswith('object_part_') or mesh.get('name') == 'object_visual':
            mesh.set('scale', vector([grasp['scale']]*3))
    for geom in xml.findall('.//geom'):
        geom.set('friction', f'{friction} 0.005 0.0001')
    key = f'candidate_{index:03d}_mu{friction:g}_{case}_{controller_mode}'
    xml_path = output/f'{key}.xml'
    xml_path.write_text(ET.tostring(xml, encoding='unicode'))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'object')
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wrist')
    mocap = model.body_mocapid[wid]
    for name, value in projected.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        data.ctrl[a] = projected[name]
    mujoco.mj_forward(model, data)
    object_vertices = []
    for geom in np.flatnonzero(model.geom_bodyid == oid):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        mesh_id = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        object_vertices.append(model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3,3).T
                               + data.geom_xpos[geom])
    object_extents = np.ptp(np.concatenate(object_vertices), axis=0)
    controller = MujocoJointCompliance(model, data, robot, projected,
                                       controller_mode, compliance_config)
    if display is not None:
        display.open(model, data)
    initial_relative = data.xpos[oid]-data.mocap_pos[mocap]
    initial_quat = data.xquat[oid].copy()
    gravity_vectors = {'gx+': [9.81,0,0], 'gx-': [-9.81,0,0],
                       'gy+': [0,9.81,0], 'gy-': [0,-9.81,0],
                       'gz+': [0,0,9.81], 'gz-': [0,0,-9.81],
                       'lift': [0,0,-9.81]}
    trace, states, mocaps, joint_trace = [], [], [], []
    gains_k, gains_d, references = [], [], []
    max_slide = max_rotation = max_torque = 0.
    max_contact_force = normal_impulse = max_target_offset = 0.
    stiffness_min, stiffness_max = float('inf'), 0.
    damping_min, damping_max = float('inf'), 0.
    initial_depth = max([max(0., -c.dist) for c in data.contact
                         if oid in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])] or [0.])
    peak_depth = initial_depth
    settle_time = .3
    steps = round((seconds+settle_time)/model.opt.timestep)
    for step in range(steps):
        if display is not None:
            display.before_step()
            # Interactive viewing must not introduce unrecorded support forces.
            data.xfrc_applied[:] = 0
            data.qfrc_applied[:] = 0
        t = step*model.opt.timestep
        phase = max(0., t-settle_time)
        model.opt.gravity[:] = np.array(gravity_vectors[case])*min(1., phase/.2)
        if case == 'lift':
            s = np.clip(phase/1., 0., 1.)
            data.mocap_pos[mocap] = position + np.array([0,0,.1*(3*s*s-2*s*s*s)])
        control = controller.step(model.opt.timestep)
        stiffness_min = min(stiffness_min, float(control['stiffness'].min()))
        stiffness_max = max(stiffness_max, float(control['stiffness'].max()))
        damping_min = min(damping_min, float(control['damping'].min()))
        damping_max = max(damping_max, float(control['damping'].max()))
        max_target_offset = max(max_target_offset, float(abs(control['reference']-controller.fixed_reference).max()))
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        peak_depth = max(peak_depth, max([max(0., -c.dist) for c in data.contact
                        if oid in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])] or [0.]))
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
            raise RuntimeError(f'Unstable simulation in {key}')
        relative = data.xpos[oid]-data.mocap_pos[mocap]
        slide = float(np.linalg.norm(relative-initial_relative))
        angle = float(2*np.arccos(np.clip(abs(np.dot(data.xquat[oid], initial_quat)),0,1)))
        max_slide, max_rotation = max(max_slide, slide), max(max_rotation, angle)
        max_torque = max(max_torque, float(np.abs(data.actuator_force).max()))
        _, contacts, force = controller.contacts()
        max_contact_force = max(max_contact_force, force)
        normal_impulse += force*model.opt.timestep
        if display is not None:
            display.after_step(model.opt.timestep)
        if step % 20 == 0 or step == steps-1:
            trace.append([float(data.time), *data.xpos[oid], *data.mocap_pos[mocap],
                          slide, np.degrees(angle), contacts, force, max_torque])
            states.append(data.qpos.copy())
            mocaps.append(data.mocap_pos.copy())
            gains_k.append(control['stiffness'])
            gains_d.append(control['damping'])
            references.append(control['reference'])
            for a, name in enumerate(controller.names):
                joint_trace.append([float(data.time), name, float(data.qpos[controller.qadr[a]]),
                                    float(data.qvel[controller.vadr[a]]), float(control['reference'][a]),
                                    float(control['stiffness'][a]), float(control['damping'][a]),
                                    float(control['filtered_force'][a]), float(data.actuator_force[a])])
    passed = max_slide < .005 and max_rotation < np.radians(15) and trace[-1][-3] > 0
    with (output/f'{key}.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['time_s','object_x','object_y','object_z','wrist_x','wrist_y','wrist_z',
                         'relative_slip_m','object_rotation_deg','object_contact_count',
                         'total_object_normal_force_N','max_actuator_torque_Nm'])
        writer.writerows(trace)
    np.savez_compressed(output/f'{key}.npz', qpos=states, mocap_pos=mocaps,
                        time=np.array(trace)[:,0], mocap_quat=data.mocap_quat.copy(),
                        stiffness=gains_k, damping=gains_d, reference=references,
                        joint_names=controller.names)
    with (output/f'{key}_joints.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['time_s','joint','q_rad','qd_rad_s','reference_rad',
                         'stiffness_Nm_rad','damping_Nm_s_rad','filtered_finger_normal_force_N','actuator_torque_Nm'])
        writer.writerows(joint_trace)
    result = {'candidate': index, 'friction': friction, 'case': case,
              'source_candidate_index': int(grasp.get('source_index',index)),
              'object_scale': float(grasp['scale']),
              'object_extents_m': object_extents.tolist(),
              'controller': controller_mode,
              'projected_pose_pass': bool(passed), 'max_relative_slip_mm': max_slide*1000,
              'source_grasp_valid': bool(passed and feasibility['source_pose_feasible']),
              'max_rotation_deg': float(np.degrees(max_rotation)),
              'final_loaded_contacts': int(trace[-1][-3]),
              'max_actuator_torque_Nm': max_torque,
              'peak_total_object_normal_force_N': max_contact_force,
              'total_normal_impulse_Ns': normal_impulse,
              'stiffness_range_Nm_rad': [stiffness_min, stiffness_max],
              'damping_range_Nm_s_rad': [damping_min, damping_max],
              'max_equilibrium_offset_rad': max_target_offset,
              'object_mass_kg': float(model.body_mass[oid]),
              'initial_max_contact_depth_mm': initial_depth*1000,
              'peak_object_contact_depth_mm': peak_depth*1000,
              'penetration_under_1mm': bool(peak_depth <= .001),
              'feasibility': feasibility, 'scene': xml_path.name}
    print(f'{key}: {"PASS" if passed else "FAIL"}, slip={max_slide*1000:.2f} mm, rotation={np.degrees(max_rotation):.1f} deg', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-file', type=Path, default=Path('data/experiments/l20_grasp_20261029/results/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03.npy'))
    parser.add_argument('--indices', type=int, nargs='+', help='Saved file indices; default picks its best candidate')
    parser.add_argument('--scale', type=float, help='Filter saved candidates by scale; never rescale a saved pose')
    parser.add_argument('--friction', type=float, nargs='+', default=[1.,3.])
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--self-collision', action='store_true', help='Enable hand self collisions; CAD convex hull overlaps can cause large internal forces')
    parser.add_argument('--controller', choices=['fixed','adaptive'], nargs='+', default=['adaptive'])
    parser.add_argument('--cases', nargs='+', choices=['gx+','gx-','gy+','gy-','gz+','gz-','lift'],
                        default=['gx+','gx-','gy+','gy-','gz+','gz-','lift'])
    parser.add_argument('--compliance-config', type=Path, help='JSON overrides for ComplianceConfig')
    parser.add_argument('--target-force', type=float, help='Desired normal contact load per finger, N')
    parser.add_argument('--output', type=Path, default=Path('data/experiments/l20_grasp_20261029/dynamics_compliance'))
    parser.add_argument('--viewer', action='store_true', help='Open live MuJoCo viewer for one test scenario')
    parser.add_argument('--hand-mesh', choices=['collision','visual','both','none'], default='collision')
    parser.add_argument('--object-mesh', choices=['collision','visual','both','none'], default='collision')
    parser.add_argument('--speed', type=float, default=1., help='Viewer wall-clock speed, e.g. 0.25 for slow motion')
    parser.add_argument('--start-paused', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--keep-open', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--show-contacts', action='store_true')
    args = parser.parse_args()
    config = ComplianceConfig(**(json.loads(args.compliance_config.read_text()) if args.compliance_config else {}))
    if args.target_force is not None:
        config = replace(config, force_target=args.target_force)
    if args.seconds <= 0 or not np.isfinite(args.seconds) or any(f <= 0 or not np.isfinite(f) for f in args.friction):
        parser.error('Duration and friction must be finite and positive')
    if len(set(args.controller)) != len(args.controller) or len(set(args.cases)) != len(args.cases):
        parser.error('Controller modes and cases must not contain duplicates')
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    if args.scale is not None and (not np.isfinite(args.scale) or args.scale <= 0):
        parser.error('--scale must be finite and positive')
    if args.viewer and len(args.indices or [0])*len(args.friction)*len(args.controller)*len(args.cases) != 1:
        parser.error('Viewer requires one scenario; use --indices 16 --friction 3 --controller adaptive --cases lift')
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        parser.error(f'Output already exists: {output}; choose a new directory')
    data = np.load(args.result_file.resolve(), allow_pickle=True)
    verify_mesh_identity(args.result_file, root/'data/meshdata'/args.result_file.stem/'coacd/decomposed.obj')
    eligible = [i for i,g in enumerate(data) if args.scale is None or np.isclose(g['scale'],args.scale,rtol=1e-6,atol=1e-9)]
    if not eligible:
        parser.error('No saved grasp has the requested scale')
    if args.indices is None:
        valid = [i for i in eligible if data[i].get('E_fc',np.inf)<.3
                 and data[i].get('E_dis',np.inf)<.005 and data[i].get('E_pen',np.inf)<.001]
        args.indices = [min(valid or eligible,key=lambda i:data[i].get('energy',np.inf))]
        print(f'Automatically selected candidate {args.indices[0]}, saved scale={data[args.indices[0]]["scale"]:.6g}',flush=True)
    if any(i not in eligible for i in args.indices):
        parser.error('Requested index is outside the file or does not match --scale')
    output.mkdir(parents=True)
    scene, robot, pieces = build_template(root, output, args.result_file.stem, args.self_collision)
    results = []
    display = None
    if args.viewer:
        from grasp_live_viewer import LiveGraspViewer, ViewerClosed
        display = LiveGraspViewer(args.hand_mesh, args.object_mesh, args.speed,
                                  args.start_paused, args.show_contacts, args.keep_open)
    try:
        for index in args.indices:
            for friction in args.friction:
                for mode in args.controller:
                    for case in args.cases:
                        results.append(run_trial(scene, robot, data[index], index, friction, case,
                                                  output, args.seconds, mode, config, display))
                        (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    except BaseException as error:
        if display is not None:
            display.close()
            if isinstance(error, (ViewerClosed, KeyboardInterrupt)):
                (output/'cancelled.json').write_text(json.dumps({'status': 'cancelled', 'reason': str(error)})+'\n')
                parser.exit(130, 'Viewer test cancelled before completion; no PASS result.\n')
        raise
    settings = {'engine': 'MuJoCo', 'engine_version': mujoco.__version__,
                'source_file': str(args.result_file.resolve()), 'object_convex_parts': pieces,
                'density_kg_m3': 500, 'friction_coefficients': args.friction,
                'gravity_m_s2': 9.81, 'loaded_duration_s': args.seconds,
                'zero_gravity_settle_s': .3, 'gravity_ramp_s': .2,
                'timestep_s': .001, 'lift_height_m': .1, 'lift_duration_s': 1.,
                'controller_kp_Nm_rad': 5., 'controller_kd_Nm_s_rad': .05,
                'controller_modes': args.controller, 'cases': args.cases,
                'mesh_display': {'viewer': args.viewer, 'hand': args.hand_mesh,
                                 'object': args.object_mesh, 'collision_convex_hulls': True,
                                 'hand_collision_group': 3, 'object_collision_group': 4},
                'adaptive_compliance': asdict(config),
                'contact_feedback': 'Sum of object normal contact loads per finger, shared by independent joints of that finger; palm loads excluded from finger feedback',
                'adaptive_damping': 'Bounded damping ratio law using diagonal of T.T @ M(q) @ T with passive mimic inertia reflected',
                'effort_limits': 'URDF per-joint effort; mimic joints are passive',
                'hand_self_collision': args.self_collision,
                'pass_limits': {'max_relative_slip_mm': 5, 'max_object_rotation_deg': 15, 'requires_final_loaded_contact': True},
                'object_support': 'none; free body, no weld, plane, adhesion, or applied support force',
                'pose_policy': 'Project saved joints to URDF limits and all five mimic equations; original feasibility reported separately',
                'limitations': 'Material friction, density and motor gains are model assumptions, not hardware calibration. This is not an Isaac Gym reproduction.'}
    (output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(f'Results: {output}', flush=True)
    if display is not None:
        try:
            display.finish()
        finally:
            display.close()


if __name__ == '__main__':
    main()
