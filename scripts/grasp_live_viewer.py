"""Passive GUI observer for the actual grasp stability simulation loop."""

from queue import SimpleQueue
from pathlib import Path
import sys
import time
import atexit
import threading

import glfw
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'grasp_generation'))
from utils.collision_display import MESH_MODES, mesh_counts, prepare_mesh_groups, set_mesh_visibility


class ViewerClosed(Exception):
    pass


class LiveGraspViewer:
    def __init__(self, hand='collision', object_mesh='collision', speed=1., paused=True,
                 contacts=False, hold=True):
        self.hand, self.object_mesh, self.speed = hand, object_mesh, speed
        self.paused, self.contacts, self.hold = paused, contacts, hold
        self.keys = SimpleQueue()
        self.handle = None
        self.last_sync = 0.
        self.viewer_threads = []
        atexit.register(self.close)

    def open(self, model, data):
        groups = prepare_mesh_groups(model)
        for category, mode in [('hand', self.hand), ('object', self.object_mesh)]:
            if mode in ('visual', 'both') and not groups[category+'_visual']:
                raise ValueError(f'{category} has no visual mesh; regenerate scene or use collision mode')
        print('Mesh counts:', mesh_counts(groups), flush=True)
        existing_threads = set(threading.enumerate())
        self.handle = mujoco.viewer.launch_passive(model, data, key_callback=self.keys.put,
                                                  show_left_ui=False, show_right_ui=False)
        # MuJoCo's public close() requests exit but does not join its daemon
        # render thread. Keep Python alive until GLFW/OpenGL teardown completes.
        self.viewer_threads = [t for t in threading.enumerate()
                               if t not in existing_threads and '_launch_internal' in t.name]
        with self.handle.lock():
            self.handle.cam.lookat[:] = np.median(data.xpos[1:], axis=0)
            self.handle.cam.distance = .48
            self.handle.cam.azimuth, self.handle.cam.elevation = -45, -20
            set_mesh_visibility(self.handle.opt, self.hand, self.object_mesh, self.contacts)
        print('F5: hand mesh | F6: cup mesh | F7: contacts | F8: pause/resume', flush=True)
        print(f'hand={self.hand}, cup={self.object_mesh}, speed={self.speed:g}x, paused={self.paused}', flush=True)
        self.handle.sync()

    def _events(self):
        changed = False
        while not self.keys.empty():
            key = self.keys.get()
            if key == glfw.KEY_F5:
                self.hand = MESH_MODES[(MESH_MODES.index(self.hand)+1)%len(MESH_MODES)]
                changed = True
            elif key == glfw.KEY_F6:
                self.object_mesh = MESH_MODES[(MESH_MODES.index(self.object_mesh)+1)%len(MESH_MODES)]
                changed = True
            elif key == glfw.KEY_F7:
                self.contacts = not self.contacts
                changed = True
            elif key == glfw.KEY_F8:
                self.paused = not self.paused
                print('Paused' if self.paused else 'Running', flush=True)
        if changed:
            with self.handle.lock():
                set_mesh_visibility(self.handle.opt, self.hand, self.object_mesh, self.contacts)
            print(f'hand={self.hand}, cup={self.object_mesh}, contacts={self.contacts}', flush=True)

    def _sync(self, force=False):
        if force or time.monotonic()-self.last_sync >= 1/60:
            self.handle.sync()
            self.last_sync = time.monotonic()

    def before_step(self):
        while self.handle.is_running():
            self._events()
            self._sync()
            if not self.paused:
                self.step_start = time.monotonic()
                return
            time.sleep(.01)
        raise ViewerClosed('Window closed before completion; test cancelled, no PASS result')

    def after_step(self, dt):
        self._sync()
        delay = self.step_start+dt/self.speed-time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def finish(self):
        if self.handle is None:
            return
        print('Test complete; results saved.' + (' Close the window to exit.' if self.hold else ''), flush=True)
        self._sync(force=True)
        while self.hold and self.handle.is_running():
            self._events()
            self._sync()
            time.sleep(.01)

    def close(self):
        if self.handle is not None:
            if self.handle.is_running():
                self.handle.close()
            for thread in self.viewer_threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=10)
            self.handle = None
