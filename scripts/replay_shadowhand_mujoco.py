"""Replay a recorded ShadowHand grasp in the native MuJoCo viewer."""

import argparse
from pathlib import Path
from queue import SimpleQueue
import sys
import time
import threading

import glfw
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grasp_generation"))
from utils.collision_display import MESH_MODES, mesh_counts, prepare_mesh_groups, set_mesh_visibility


DEFAULT_RESULTS = Path("data/experiments/shadow_cup_jenga_006_all_mujoco_20261102")
OBJECTS = {
    "mug": "core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03",
    "jenga": "2_of_Jenga_Classic_Game",
}
CASES = ("gx+", "gx-", "gy+", "gy-", "gz+", "gz-", "lift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--protocol", choices=("strict", "gym"), default="strict")
    parser.add_argument("--variant", choices=("official", "no-force"), default="official")
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--candidate", type=int, choices=range(16), metavar="0..15", required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--hand-mesh", choices=MESH_MODES, default="collision")
    parser.add_argument("--object-mesh", choices=MESH_MODES, default="collision")
    parser.add_argument("--show-contacts", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    gym_protocol = args.protocol == "gym"
    args.case = args.case or ("gy-" if gym_protocol else "lift")
    if gym_protocol:
        if args.results is None or args.case == 'lift':
            parser.error('Gym protocol requires --results and supports six gravity cases without lift')
        folder = args.results / args.object / args.variant / f"candidate_{args.candidate:04d}"
        scene, recording = folder / 'scene.xml', folder / 'trajectory.npz'
    else:
        folder = (args.results or DEFAULT_RESULTS) / OBJECTS[args.object] / f"candidate_{args.candidate:04d}"
        scene = folder / f"scene_{args.case}_cpu.xml"
        recording = folder / f"official_{args.case}_cpu.npz"
    if not scene.is_file() or not recording.is_file():
        parser.error(f"MuJoCo scene or trajectory missing in {folder}")

    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    object_names = [f'd{i}:object' for i in range(6)] if gym_protocol else ['object']
    groups = prepare_mesh_groups(model, object_names)
    print("Collision/visual geoms:", mesh_counts(groups), flush=True)
    with np.load(recording) as recording_data:
        states = {key: recording_data[key] for key in
                  ("time", "qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat") if key in recording_data}
    data = mujoco.MjData(model)

    def set_frame(index):
        data.qpos[:] = states["qpos"][index]
        data.qvel[:] = states["qvel"][index]
        data.ctrl[:] = states["ctrl"][index]
        if 'mocap_pos' in states:
            data.mocap_pos[:] = states["mocap_pos"][index]
            data.mocap_quat[:] = states["mocap_quat"][index]
        data.time = float(states["time"][index])
        mujoco.mj_forward(model, data)

    set_frame(0)
    focus_positions = [data.xpos[model.body(name).id].copy() for name in object_names]
    events = SimpleQueue()
    existing_threads = set(threading.enumerate())
    with mujoco.viewer.launch_passive(model, data, key_callback=events.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        render_threads = [t for t in threading.enumerate() if t not in existing_threads
                          and '_launch_internal' in t.name]
        with viewer.lock():
            if gym_protocol:
                direction = ("gy-", "gy+", "gx-", "gx+", "gz+", "gz-").index(args.case)
                viewer.cam.lookat[:] = focus_positions[direction]
            else:
                viewer.cam.lookat[:] = (data.xpos[model.body("object").id] +
                                        data.xpos[model.body("wrist").id]) / 2
            viewer.cam.distance = 0.35
            viewer.cam.azimuth, viewer.cam.elevation = -45, -20
            set_mesh_visibility(viewer.opt, args.hand_mesh, args.object_mesh, args.show_contacts)
        print("F5: hand geoms | F6: object geoms | F7: contacts | SPACE/F8: pause/resume", flush=True)
        print("Left/Right: step one recorded frame while paused; close window to exit", flush=True)
        if gym_protocol:
            print('1..6: focus gravity scene (gy-, gy+, gx-, gx+, gz+, gz-); native mouse controls rotate/pan/zoom', flush=True)
        print(f"Replay: {args.object} #{args.candidate} {args.case}; {len(states['time'])} frames", flush=True)
        index = 0
        paused = args.paused
        hand, obj, contacts = args.hand_mesh, args.object_mesh, args.show_contacts
        while viewer.is_running():
            while not events.empty():
                key = events.get()
                if key == glfw.KEY_F5:
                    hand = MESH_MODES[(MESH_MODES.index(hand) + 1) % len(MESH_MODES)]
                elif key == glfw.KEY_F6:
                    obj = MESH_MODES[(MESH_MODES.index(obj) + 1) % len(MESH_MODES)]
                elif key == glfw.KEY_F7:
                    contacts = not contacts
                elif key in (glfw.KEY_F8, glfw.KEY_SPACE):
                    paused = not paused
                elif gym_protocol and ord('1') <= key <= ord('6'):
                    with viewer.lock():
                        viewer.cam.lookat[:] = focus_positions[key - ord('1')]
                elif paused and key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                    index = max(0, min(len(states["time"]) - 1,
                                       index + (1 if key == glfw.KEY_RIGHT else -1)))
                with viewer.lock():
                    set_mesh_visibility(viewer.opt, hand, obj, contacts)
            start = time.monotonic()
            with viewer.lock():
                set_frame(index)
            viewer.sync()
            if paused:
                time.sleep(0.02)
                continue
            if index == len(states["time"]) - 1:
                if args.no_loop:
                    break
                index = 0
            else:
                index += 1
            dt = max(0.0, float(states["time"][index] - states["time"][index - 1])) if index else 0.04
            time.sleep(max(0.0, dt / args.speed - (time.monotonic() - start)))
    for thread in render_threads:
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
