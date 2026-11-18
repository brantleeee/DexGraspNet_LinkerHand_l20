"""Render a recorded MuJoCo validation trajectory; simulation is not rerun."""

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'grasp_generation'))
from utils.collision_display import MESH_MODES, mesh_counts, prepare_mesh_groups, set_mesh_visibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--trajectory', type=Path, help='Explicit NPZ for shared-model mjlab batch recordings')
    parser.add_argument('--hand-mesh', choices=MESH_MODES, default='collision')
    parser.add_argument('--object-mesh', choices=MESH_MODES, default='collision')
    parser.add_argument('--show-contacts', action='store_true')
    parser.add_argument('--object-body', help='Camera target body; defaults to object or d0:object')
    parser.add_argument('--speed', type=float, default=0.5, help='Playback speed, simulated seconds per video second')
    parser.add_argument('--fps', type=int, default=25)
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0 or args.fps <= 0:
        parser.error('speed/fps must be positive and finite')
    scene = args.scene.resolve()
    output = args.output.resolve() if args.output else scene.with_suffix('.mp4')
    trajectory = np.load(args.trajectory or scene.with_suffix('.npz'))
    model = mujoco.MjModel.from_xml_path(str(scene))
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    object_names = ['object'] if 'object' in body_names else [name for name in body_names if name and name.endswith(':object')]
    groups = prepare_mesh_groups(model, object_names)
    object_body = args.object_body or object_names[0]
    if object_body not in body_names:
        parser.error('camera target body does not exist')
    print('Mesh counts:', mesh_counts(groups))
    for category, mode in [('hand', args.hand_mesh), ('object', args.object_mesh)]:
        if mode in ('visual', 'both') and not groups[category+'_visual']:
            parser.error(f'{category} visual mesh is missing; regenerate scene or choose collision')
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    data.qpos[:] = trajectory['qpos'][0]
    quaternions = trajectory['mocap_quat'] if 'mocap_quat' in trajectory else None
    if quaternions is not None:
        data.mocap_pos[:] = trajectory['mocap_pos'][0]
        data.mocap_quat[:] = quaternions[0] if quaternions.ndim == 3 else quaternions
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = data.xpos[model.body(object_body).id]
    camera.distance, camera.azimuth, camera.elevation = .48, -45, -20
    option = mujoco.MjvOption()
    set_mesh_visibility(option, args.hand_mesh, args.object_mesh, args.show_contacts)
    with mujoco.Renderer(model, 720, 960) as renderer:
        samples = np.arange(trajectory['time'][0], trajectory['time'][-1] + 1e-9, args.speed / args.fps)
        indices = np.clip(np.searchsorted(trajectory['time'], samples), 0, len(trajectory['time']) - 1)
        release_frame = int(np.argmin(abs(samples - .4)))
        with imageio.get_writer(output, fps=args.fps, codec='libx264', quality=8) as writer:
            for frame_index, i in enumerate(indices):
                data.qpos[:] = trajectory['qpos'][i]
                if quaternions is not None:
                    data.mocap_pos[:] = trajectory['mocap_pos'][i]
                    data.mocap_quat[:] = quaternions[i] if quaternions.ndim == 3 else quaternions
                if 'qvel' in trajectory:
                    data.qvel[:] = trajectory['qvel'][i]
                if 'ctrl' in trajectory:
                    data.ctrl[:] = trajectory['ctrl'][i]
                data.time = trajectory['time'][i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = renderer.render()
                writer.append_data(frame)
                if frame_index == 0:
                    imageio.imwrite(output.with_name(output.stem+'_initial.png'), frame)
                if frame_index == release_frame:
                    imageio.imwrite(output.with_name(output.stem+'_released.png'), frame)
    print(f'Video ({args.speed:g}x playback): {output}')


if __name__ == '__main__':
    main()
