"""Check whether translating a mocap collider supplies tangential velocity.

A passive box rests on a moving, high-friction platform. Compare mocap pose
updates with a prescribed slide-joint position AND analytic velocity. This is a
protocol diagnostic, not a grasp success test.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def trial(mode, output):
    moving = 'mocap="true"' if mode == "mocap" else ''
    joint = '' if mode == "mocap" else '<joint name="platform_slide" type="slide" axis="1 0 0" armature="10000"/>'
    xml = f'''<mujoco><option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81" iterations="100"/>
    <default><geom friction="3 0.005 0.0001" condim="3" solref="0.005 1"/></default>
    <worldbody><body name="platform" {moving}>{joint}<geom type="box" size="0.3 0.3 0.02" mass="1000"/></body>
    <body name="object" pos="0 0 0.051"><freejoint name="object_free"/><geom type="box" size="0.03 0.03 0.03" mass="0.1"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    platform, obj = model.body("platform").id, model.body("object").id
    trace = []
    for step in range(1500):
        t = step * .001
        u = np.clip((t-.3)/1., 0., 1.)
        position = .1*(3*u*u-2*u*u*u)
        velocity = .6*u*(1-u)
        if mode == "mocap":
            data.mocap_pos[model.body_mocapid[platform], 0] = position
        else:
            jid = model.joint("platform_slide").id
            data.qpos[model.jnt_qposadr[jid]] = position
            data.qvel[model.jnt_dofadr[jid]] = velocity
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        trace.append([data.time, position, velocity, *data.cvel[platform, 3:], *data.xpos[obj]])
    trace = np.array(trace)
    np.savetxt(output/f"{mode}.csv", trace, delimiter=",", header="time,command_x,command_vx,solver_vx,solver_vy,solver_vz,object_x,object_y,object_z", comments="")
    return {"mode": mode, "platform_displacement_m": float(trace[-1,1]), "object_displacement_m": float(trace[-1,6]),
            "peak_commanded_velocity_m_s": float(trace[:,2].max()), "peak_solver_platform_velocity_m_s": float(abs(trace[:,3]).max()),
            "final_relative_slip_m": float(abs(trace[-1,1]-trace[-1,6]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = [trial(mode, args.output) for mode in ["mocap", "slide_with_velocity"]]
    (args.output/"results.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
