"""Candidate import/export shared by deterministic refinement and RL references."""

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def vector(values):
    return " ".join(f"{float(x):.12g}" for x in values)


def load_candidate(path, index, data=None):
    if data is None:
        data = np.load(path, allow_pickle=True)
    if not 0 <= index < len(data):
        raise ValueError(f"Candidate {index} outside file containing {len(data)} entries")
    grasp = copy.deepcopy(data[index])
    values = [grasp["scale"], *grasp["qpos"].values()]
    if not np.isfinite(values).all() or grasp["scale"] <= 0:
        raise ValueError("Candidate contains nonfinite values or an invalid scale")
    return grasp


def configure_scene(template, grasp, friction):
    """Set wrist and object scale from ONE candidate; no pose rescaling."""
    scene = copy.deepcopy(template)
    q = grasp["qpos"]
    wrist = scene.find("worldbody/body[@name='wrist']")
    wrist.set("pos", vector([q["WRJT" + a] for a in "xyz"]))
    quat = Rotation.from_euler("xyz", [q["WRJR" + a] for a in "xyz"]).as_quat()
    wrist.set("quat", vector(quat[[3, 0, 1, 2]]))
    for mesh in scene.findall("asset/mesh"):
        if mesh.get("name", "").startswith("object_"):
            mesh.set("scale", vector([grasp["scale"]] * 3))
    for geom in scene.findall(".//geom"):
        geom.set("friction", f"{friction:g} 0.005 0.0001")
    return scene


def wrist_pose(grasp):
    q = grasp["qpos"]
    position = [q["WRJT" + a] for a in "xyz"]
    quat = Rotation.from_euler("xyz", [q["WRJR" + a] for a in "xyz"]).as_quat()[[3, 0, 1, 2]]
    return position, quat


def export_reference(result_file, index, scene, output, grasp=None, provenance=None):
    """Split a configured scene into assets and candidate state, without simulation."""
    grasp = load_candidate(result_file, index) if grasp is None else grasp
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference: {output}")
    hand = copy.deepcopy(scene)
    for tag in ("sensor", "keyframe"):
        element = hand.find(tag)
        if element is not None:
            hand.remove(element)
    option = hand.find("option")
    if option is not None:
        hand.remove(option)
    hand.find("worldbody").remove(hand.find("worldbody/body[@name='object']"))
    for light in hand.findall("worldbody/light"):
        hand.find("worldbody").remove(light)
    for mesh in list(hand.find("asset")):
        if mesh.get("name", "").startswith("object_"):
            hand.find("asset").remove(mesh)
    obj = ET.Element("mujoco", model="grasp_object")
    ET.SubElement(obj, "compiler", angle="radian")
    asset = ET.SubElement(obj, "asset")
    for mesh in scene.find("asset"):
        if mesh.get("name", "").startswith("object_"):
            asset.append(copy.deepcopy(mesh))
    ET.SubElement(obj, "worldbody").append(copy.deepcopy(scene.find("worldbody/body[@name='object']")))
    joints = [j.get("name") for j in hand.findall(".//joint") if j.get("axis") is not None]
    drives = [a.get("joint") for a in hand.find("actuator")]
    if (len(joints), len(drives), len(hand.findall("equality/joint"))) != (21, 16, 5):
        raise ValueError("Unexpected L20 joint/mimic structure")
    bounds = {a.get("joint"): list(map(float, a.get("ctrlrange").split())) for a in hand.find("actuator")}
    for name in drives:
        if not bounds[name][0] - 1e-6 <= grasp["qpos"][name] <= bounds[name][1] + 1e-6:
            raise ValueError(f"Infeasible candidate joint: {name}")
    for joint in hand.findall("worldbody/.//joint"):
        name = joint.get("name")
        if name not in joints:
            continue
        value = grasp["qpos"][name]
        lo, hi = map(float, joint.get("range").split())
        if not np.isfinite(value) or not lo - 1e-6 <= value <= hi + 1e-6:
            raise ValueError(f"Infeasible candidate joint: {name}")
    for equality in hand.findall("equality/joint"):
        child, parent = equality.get("joint1"), equality.get("joint2")
        coefficients = map(float, equality.get("polycoef", "0 1 0 0 0").split())
        expected = sum(c * grasp["qpos"][parent] ** i for i, c in enumerate(coefficients))
        if abs(grasp["qpos"][child] - expected) > 1e-4:
            raise ValueError(f"Infeasible candidate mimic: {child}")
    wrist = scene.find("worldbody/body[@name='wrist']")
    object_body = scene.find("worldbody/body[@name='object']")
    ref = {
        "schema_version": 2, "scale": float(grasp["scale"]),
        "object_code": grasp.get("object_code", Path(result_file).stem),
        "source_file": str(Path(result_file).resolve()), "source_index": index,
        "source_batch_index": int(grasp.get("source_index", index)),
        "joint_names": joints, "drive_names": drives,
        "joint_pos": {n: float(grasp["qpos"][n]) for n in joints},
        "drive_limits": bounds,
        "wrist_pos": list(map(float, wrist.get("pos").split())),
        "wrist_quat": list(map(float, wrist.get("quat").split())),
        "object_pose": list(map(float, object_body.get("pos", "0 0 0").split()))
        + list(map(float, object_body.get("quat", "1 0 0 0").split())),
        "action_zero": "q_target = q_static + 0.2 * clip(action, -1, 1), then limit clipping",
        "provenance": provenance or {"variant": "original"},
        "note": "Candidate reference; physical success must be checked separately.",
    }
    # Compile before creating output, so invalid XML cannot look like a valid reference.
    xmls = {name: ET.tostring(element, encoding="unicode") for name, element in [("hand.xml", hand), ("object.xml", obj)]}
    for xml in xmls.values():
        mujoco.MjModel.from_xml_string(xml)
    output.mkdir(parents=True)
    for name, xml in xmls.items():
        (output / name).write_text(xml)
    ref["asset_sha256"] = {n: hashlib.sha256(x.encode()).hexdigest() for n, x in xmls.items()}
    (output / "reference.json").write_text(json.dumps(ref, indent=2) + "\n")
    return ref


def initialize_data(model, grasps):
    """Initialize full state by joint names (never assume object/hand qpos order)."""
    states, controls = [], []
    for grasp in grasps:
        data = mujoco.MjData(model)
        for name, value in grasp["qpos"].items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = value
        for a, jid in enumerate(model.actuator_trnid[:, 0]):
            data.ctrl[a] = grasp["qpos"][model.joint(jid).name]
        mujoco.mj_forward(model, data)
        states.append(data.qpos.copy())
        controls.append(data.ctrl.copy())
    return np.array(states), np.array(controls)
