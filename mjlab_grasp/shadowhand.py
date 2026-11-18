"""DexGraspNet's pinned ShadowHand assets, without L20 pose conversion.

The source file has a fixed palm and dangling WRJ0/WRJ1 references. Only
those references and obsolete MuJoCo options are removed. Finger mechanics,
inertias, actuator gains/force limits, and limited tendons are retained.
"""

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

from mjlab_grasp.candidates import ROOT, configure_scene


OFFICIAL = ROOT / "assets/shadowhand_official"


def verify_official_assets():
    manifest = json.loads((OFFICIAL / "SOURCE.json").read_text())
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((OFFICIAL / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Official asset hash mismatch: {name}")
    return manifest


def build_shadow_template(object_code, output, physics_profile="source", object_mesh=None):
    if physics_profile not in ("source", "asset-defaults"):
        raise ValueError("Unknown ShadowHand physics profile")
    source = OFFICIAL / "grasp_generation/mjcf"
    scene = ET.parse(source / "shadow_hand_wrist_free.xml").getroot()
    scene.set("model", "dexgraspnet_shadowhand_mujoco")
    compiler = scene.find("compiler")
    compiler.attrib.pop("meshdir", None)
    compiler.attrib.pop("texturedir", None)
    scene.find("option").attrib.pop("apirate", None)
    # nstack was superseded by dynamic memory allocation in MuJoCo 3.
    scene.find("size").attrib.pop("nstack", None)
    for mesh in scene.findall("asset/mesh"):
        mesh.set("file", str((source / "meshes" / mesh.get("file")).resolve()))
    world = scene.find("worldbody")
    palm = world.find("body[@name='robot0:palm']")
    if physics_profile == "asset-defaults":
        # The generation XML lost the ancestor carrying this childclass.
        # Opt-in: activate upstream's declared physics rather than invent gains.
        palm.set("childclass", "robot0:asset_class")
    names = {joint.get("name") for joint in palm.iter("joint")}
    removed = []
    for tag in ("actuator", "sensor"):
        for element in list(scene.find(tag)):
            joint = element.get("joint")
            if joint is not None and joint not in names:
                if joint not in {"robot0:WRJ0", "robot0:WRJ1"}:
                    raise ValueError(f"Unexpected dangling official joint: {joint}")
                removed.append(element.get("name"))
                scene.find(tag).remove(element)
    world.remove(palm)
    ET.SubElement(world, "body", name="wrist", mocap="true").append(palm)
    # Hand vs object enabled, hand vs hand disabled, as in current L20 protocol.
    # Primitive collision geometry and native per-geom condim are preserved.
    for geom in palm.iter("geom"):
        geom.set("contype", "1")
        geom.set("conaffinity", "2")
    mesh_path = (Path(object_mesh).resolve() if object_mesh is not None else
                 OFFICIAL / "data/meshdata" / object_code / "coacd/decomposed.obj")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Object collision mesh is missing: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    pieces = mesh.split(only_watertight=False)
    asset_dir = Path(output) / "collision_parts"
    asset_dir.mkdir(parents=True, exist_ok=True)
    obj = ET.SubElement(world, "body", name="object")
    ET.SubElement(obj, "freejoint", name="object_free")
    for i, piece in enumerate(pieces):
        path = asset_dir / f"part_{i:03d}.obj"
        piece.convex_hull.export(path)
        ET.SubElement(scene.find("asset"), "mesh", name=f"object_part_{i}", file=str(path.resolve()))
        ET.SubElement(obj, "geom", name=f"object_geom_{i}", type="mesh", mesh=f"object_part_{i}",
                      density="500", contype="2", conaffinity="1", condim="3", margin="0",
                      solref="0.005 1", solimp="0.95 0.99 0.001", rgba="0.35 0.75 0.25 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
    if (len(names), model.nu, model.ntendon) != (22, 18, 4):
        raise ValueError("Unexpected official ShadowHand mechanism")
    return scene, {"finger_joints": 22, "actuators": 18, "limited_tendons": 4,
                   "object_mesh_file": str(mesh_path),
                   "object_mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                   "physics_profile": physics_profile,
                   "finger_damping": model.dof_damping[model.jnt_dofadr[model.jnt_type == 3]].tolist(),
                   "finger_armature": model.dof_armature[model.jnt_dofadr[model.jnt_type == 3]].tolist(),
                   "removed_dangling_wrist_references": removed, "object_convex_parts": len(pieces),
                   "native_actuator_gains": model.actuator_gainprm[:, 0].tolist(),
                   "self_collision": False, "object_density_kg_m3": 500,
                   "adaptations": ["remove obsolete apirate/nstack", "absolute mesh paths",
                                   "remove dangling wrist references", "ideal arm wrapper",
                                   "hand/object collision masks", "validation contact margin zero"]}


def prepare_shadow_grasp(template, grasp, friction):
    expected = {j.get("name") for j in template.findall("worldbody/.//joint")}
    expected.discard("object_free")
    received = {n for n in grasp["qpos"] if n not in {"WRJT"+a for a in "xyz"} | {"WRJR"+a for a in "xyz"}}
    if expected != received:
        raise ValueError(f"Expected native ShadowHand joint names; missing={expected-received}, extra={received-expected}")
    if not np.isfinite([grasp["scale"], *grasp["qpos"].values()]).all() or grasp["scale"] <= 0:
        raise ValueError("Invalid ShadowHand candidate")
    # Keep source joint values, including passive joints and any infeasibility.
    # Reprojection would alter the official baseline being tested.
    return configure_scene(template, copy.deepcopy(grasp), friction)
