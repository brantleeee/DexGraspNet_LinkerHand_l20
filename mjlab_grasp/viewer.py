"""Native mjlab policy viewer showing separate hand/cup collision hulls."""

import threading
import time

from mjlab.viewer import NativeMujocoViewer

from grasp_generation.utils.collision_display import (
    prepare_mesh_groups,
    set_mesh_visibility,
)


class GraspPolicyViewer(NativeMujocoViewer):
    def __init__(self, env, policy, duration=None):
        super().__init__(env, policy, enable_perturbations=False)
        self.duration = duration
        self.render_threads = []

    def setup(self):
        existing = set(threading.enumerate())
        super().setup()
        self.render_threads = [
            t
            for t in threading.enumerate()
            if t not in existing and "_launch_internal" in t.name
        ]
        prepare_mesh_groups(self.mjm, object_body_name="object/object")
        with self.viewer.lock():
            set_mesh_visibility(self.viewer.opt, "collision", "collision")
            set_mesh_visibility(self.vopt, "collision", "collision")
        self.started = time.monotonic()

    def is_running(self):
        if (
            self.duration is not None
            and time.monotonic() - self.started >= self.duration
        ):
            return False
        return super().is_running()

    def close(self):
        super().close()
        for thread in self.render_threads:
            if thread is not threading.current_thread():
                thread.join(timeout=10)
