"""Native Isaac Gym display with an environment-relative camera and pause control."""

import json
import math
import time

from isaacgym import gymapi


class GymGraspViewer:
    def __init__(self, sim, gym, output, autostart=False, hold_seconds=0, speed=1):
        self.sim, self.gym, self.output = sim, gym, output
        self.paused = not autostart
        self.hold_seconds = hold_seconds
        self.direction = 0
        self.steps = 0
        self.finished = False
        self.speed = speed
        self.dragging = False
        self.last_mouse = None
        self.reset_orbit()
        self.sim_start = gym.get_sim_time(sim.sim)
        if sim.viewer is None:
            raise RuntimeError('Isaac Gym could not create the viewer')
        gym.prepare_sim(sim.sim)
        gym.set_light_parameters(sim.sim, 0, gymapi.Vec3(0.8, 0.8, 0.8),
                                 gymapi.Vec3(0.6, 0.6, 0.6), gymapi.Vec3(-0.5, -1, -0.5))
        for env, hand, obj in zip(sim.envs, sim.hand_handles, sim.obj_handles):
            for actor, color in ((hand, gymapi.Vec3(0.2, 0.6, 0.95)),
                                 (obj, gymapi.Vec3(0.95, 0.65, 0.15))):
                for body in range(gym.get_actor_rigid_body_count(env, actor)):
                    gym.set_rigid_body_color(env, actor, body, gymapi.MESH_VISUAL_AND_COLLISION, color)
        gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_SPACE, 'pause')
        for index in range(6):
            gym.subscribe_viewer_keyboard_event(sim.viewer, getattr(gymapi, f'KEY_{index + 1}'), f'view_{index}')
        for key, action in [('LEFT', 'left'), ('RIGHT', 'right'), ('UP', 'up'), ('DOWN', 'down'),
                            ('PAGE_UP', 'zoom_in'), ('PAGE_DOWN', 'zoom_out'), ('HOME', 'home')]:
            gym.subscribe_viewer_keyboard_event(sim.viewer, getattr(gymapi, f'KEY_{key}'), action)
        gym.subscribe_viewer_mouse_event(sim.viewer, gymapi.MOUSE_LEFT_BUTTON, 'orbit_drag')
        self.focus(0)
        self.render()
        gym.write_viewer_image_to_file(sim.viewer, str(output / 'gym_initial.png'))
        print('SPACE: start/pause | 1..6: select gravity scene | Left-drag/arrows: orbit', flush=True)
        print('PageUp/PageDown: zoom | Home: reset camera. Click the window before using keys.', flush=True)
        print(f'Test: {sim.sim_step} steps x {sim.sim_params.dt:g} s = '
              f'{sim.sim_step * sim.sim_params.dt:.3f} simulated seconds; display speed={speed:g}x', flush=True)
        print('Validation starts paused.' if self.paused else 'Validation running.', flush=True)
        self.progress('paused' if self.paused else 'running')

    def reset_orbit(self):
        self.yaw = math.atan2(0.16, 0.22)
        self.pitch = math.atan2(0.10, math.hypot(0.16, 0.22))
        self.distance = math.sqrt(0.16 ** 2 + 0.10 ** 2 + 0.22 ** 2)

    def focus(self, direction):
        self.direction = direction
        self.camera()

    def camera(self):
        # Passing the actual env is essential: its origin is not world (0,0,0).
        self.pitch = max(-1.4, min(1.4, self.pitch))
        self.distance = max(0.08, min(2, self.distance))
        r = self.distance * math.cos(self.pitch)
        position = gymapi.Vec3(r * math.sin(self.yaw), self.distance * math.sin(self.pitch), r * math.cos(self.yaw))
        self.gym.viewer_camera_look_at(self.sim.viewer, self.sim.envs[self.direction], position, gymapi.Vec3(0, 0, 0))

    def progress(self, status):
        elapsed = self.gym.get_sim_time(self.sim.sim) - self.sim_start
        record = {'status': status, 'executed_steps': self.steps, 'requested_steps': self.sim.sim_step,
                  'simulated_seconds': elapsed, 'expected_seconds': self.sim.sim_step * self.sim.sim_params.dt,
                  'display_speed': self.speed}
        path = self.output / 'gui_progress.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(path)
        return elapsed

    def events(self):
        for event in self.gym.query_viewer_action_events(self.sim.viewer):
            if event.action == 'orbit_drag':
                self.dragging = event.value > 0
                self.last_mouse = self.gym.get_viewer_mouse_position(self.sim.viewer)
                continue
            if event.value <= 0:
                continue
            if event.action == 'pause':
                if self.finished:
                    print('Validation finished; camera remains interactive, physics is stopped.', flush=True)
                else:
                    self.paused = not self.paused
                    self.progress('paused' if self.paused else 'running')
            elif event.action.startswith('view_'):
                self.focus(int(event.action.split('_')[1]))
            elif event.action in ('left', 'right', 'up', 'down', 'zoom_in', 'zoom_out', 'home'):
                if event.action == 'left':
                    self.yaw -= 0.15
                elif event.action == 'right':
                    self.yaw += 0.15
                elif event.action == 'up':
                    self.pitch += 0.15
                elif event.action == 'down':
                    self.pitch -= 0.15
                elif event.action == 'zoom_in':
                    self.distance *= 0.85
                elif event.action == 'zoom_out':
                    self.distance /= 0.85
                else:
                    self.reset_orbit()
                self.camera()
        if self.dragging:
            mouse = self.gym.get_viewer_mouse_position(self.sim.viewer)
            if self.last_mouse is not None:
                self.yaw -= (mouse.x - self.last_mouse.x) * 2 * math.pi
                self.pitch += (mouse.y - self.last_mouse.y) * math.pi
                self.camera()
            self.last_mouse = mouse

    def render(self):
        self.gym.step_graphics(self.sim.sim)
        self.gym.draw_viewer(self.sim.viewer, self.sim.sim, False)

    def run(self):
        wall_start = time.monotonic()
        while self.steps < self.sim.sim_step:
            if self.gym.query_viewer_has_closed(self.sim.viewer):
                self.progress('cancelled')
                raise RuntimeError(f'Viewer closed before all {self.sim.sim_step} steps; validation cancelled')
            frame_start = time.monotonic()
            self.events()
            if not self.paused:
                self.gym.simulate(self.sim.sim)
                self.gym.fetch_results(self.sim.sim, True)
                self.steps += 1
                if self.steps % 100 == 0 or self.steps == self.sim.sim_step:
                    elapsed = self.progress('running')
                    print(f'Progress: {self.steps}/{self.sim.sim_step} steps; engine time={elapsed:.3f} s', flush=True)
            self.render()
            interval = self.sim.sim_params.dt / self.speed if not self.paused else 0.02
            time.sleep(max(0, interval - (time.monotonic() - frame_start)))
        self.wall_seconds = time.monotonic() - wall_start
        self.finished = True
        self.progress('finished')
        self.render()
        self.gym.write_viewer_image_to_file(self.sim.viewer, str(self.output / 'gym_final.png'))

    def hold(self):
        print('Validation saved. Window stays open; use 1..6 to inspect directions, close to exit.', flush=True)
        start = time.monotonic()
        while not self.gym.query_viewer_has_closed(self.sim.viewer):
            if self.hold_seconds > 0 and time.monotonic() - start >= self.hold_seconds:
                break
            self.events()
            self.render()
            time.sleep(0.02)
