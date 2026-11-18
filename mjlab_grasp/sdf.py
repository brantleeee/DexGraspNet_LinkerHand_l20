"""Opt-in mesh SDF references for MuJoCo/MJWarp 3.16; never alter old assets."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collision_geom(geom):
    return int(geom.get('contype', '1')) != 0 or int(geom.get('conaffinity', '1')) != 0


def check_reference_compatibility(reference, checkpoint_reference):
    """Legacy checkpoints remain loadable, but may never cross the SDF boundary."""
    current = reference.get('collision_representation', 'legacy_convex')
    saved = checkpoint_reference.get('collision_representation', 'legacy_convex')
    if current != saved:
        raise ValueError('Checkpoint collision representation differs; train a new SDF policy')
    if current == 'mesh_sdf' and reference.get('asset_sha256') != checkpoint_reference.get('asset_sha256'):
        raise ValueError('SDF checkpoint/reference assets differ')


def convert_xml(root, object_mesh=None, scale=1.0):
    root = copy.deepcopy(root)
    old = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    # Preserve inertial properties before changing collision shape or object assembly.
    for body in root.findall('.//worldbody//body'):
        bid = mujoco.mj_name2id(old, mujoco.mjtObj.mjOBJ_BODY, body.get('name', ''))
        if bid < 1 or old.body_mass[bid] <= 0:
            continue
        inertia = body.find('inertial')
        if inertia is not None:
            body.remove(inertia)
        fmt = lambda values: ' '.join(format(float(v), '.17g') for v in values)
        ET.SubElement(body, 'inertial', pos=fmt(old.body_ipos[bid]), quat=fmt(old.body_iquat[bid]),
                      mass=format(float(old.body_mass[bid]), '.17g'), diaginertia=fmt(old.body_inertia[bid]))
    if object_mesh is not None:
        body = root.find("worldbody/body[@name='object']")
        if body is None:
            raise ValueError('Expected object root')
        geoms = [g for g in body.findall('geom') if collision_geom(g)]
        if not geoms:
            raise ValueError('No object collision geometry')
        template = copy.deepcopy(geoms[0])
        for geom in geoms:
            body.remove(geom)
        for key in ('pos', 'quat', 'euler', 'axisangle', 'fromto', 'size', 'mass', 'density'):
            template.attrib.pop(key, None)
        template.set('name', 'object_sdf'); template.set('mesh', 'object_sdf_mesh')
        template.set('type', 'sdf'); template.set('density', '0')
        body.append(template)
        asset = root.find('asset')
        ET.SubElement(asset, 'mesh', name='object_sdf_mesh', file=str(Path(object_mesh).resolve()),
                      scale=' '.join([format(float(scale), '.17g')]*3))
    count = 0
    for geom in root.findall('.//worldbody//geom'):
        if collision_geom(geom) and geom.get('mesh'):
            geom.set('type', 'sdf'); count += 1
    if count == 0:
        raise ValueError('No mesh collision geoms converted')
    # Compile before returning: unsupported versions and invalid meshes fail explicitly.
    new = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    for name in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
        np.testing.assert_allclose(getattr(old, name), getattr(new, name), rtol=1e-9, atol=1e-12)
    return root, new, count


def export_sdf_reference(source, output, object_mesh=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    ref = json.loads((source/'reference.json').read_text())
    if ref.get('collision_representation') == 'mesh_sdf':
        raise ValueError('Source already uses SDF')
    if object_mesh is None:
        code = ref.get('object_code') or Path(ref['source_file']).stem
        object_mesh = Path(__file__).resolve().parents[1]/'data/meshdata'/code/'coacd/decomposed.obj'
    object_mesh = Path(object_mesh).resolve()
    if not object_mesh.is_file():
        raise FileNotFoundError(f'Pass --object-mesh explicitly: {object_mesh}')
    xmls, models, counts = {}, {}, {}
    for name in ('hand', 'object'):
        path = source/f'{name}.xml'
        root = ET.parse(path).getroot()
        # Freeze relative asset paths at the original XML directory.
        compiler = root.find('compiler')
        meshdir = compiler.get('meshdir', '') if compiler is not None else ''
        for mesh in root.findall('asset/mesh'):
            if mesh.get('file') and not Path(mesh.get('file')).is_absolute():
                mesh.set('file', str((source/meshdir/mesh.get('file')).resolve()))
        converted, model, count = convert_xml(root, object_mesh if name == 'object' else None, ref['scale'])
        xmls[f'{name}.xml'] = ET.tostring(converted, encoding='unicode')
        models[name], counts[name] = model, count
    ref.update(collision_representation='mesh_sdf', sdf_source_reference=str(source),
               sdf_object_source=str(object_mesh), sdf_object_sha256=digest(object_mesh),
               sdf_note='Native mesh SDF, no custom analytic plugin. Object is recovered CoACD source, not restored original geometry.')
    ref['asset_sha256'] = {k: hashlib.sha256(v.encode()).hexdigest() for k,v in xmls.items()}
    mesh_hashes = {}
    for xml in xmls.values():
        for mesh in ET.fromstring(xml).findall('asset/mesh'):
            if mesh.get('file'):
                mesh_hashes[mesh.get('file')] = digest(mesh.get('file'))
    ref['sdf_mesh_sha256'] = mesh_hashes
    audit = dict(mujoco_version=mujoco.__version__, sdf_collision_geoms=counts,
                 inertial_properties_preserved=True, source_reference=str(source),
                 mesh_sha256=mesh_hashes, gpu_execution_verified=False)
    output.mkdir(parents=True)
    for name, xml in xmls.items():
        (output/name).write_text(xml)
    (output/'reference.json').write_text(json.dumps(ref, indent=2)+'\n')
    (output/'sdf_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return audit
