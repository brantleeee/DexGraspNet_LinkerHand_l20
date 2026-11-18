"""Separate visibility for hand/object visual meshes and collision convex hulls."""

import colorsys

import mujoco


MESH_MODES = ('collision', 'visual', 'both', 'none')


def prepare_mesh_groups(model, object_body_name='object'):
    names = [object_body_name] if isinstance(object_body_name, str) else object_body_name
    object_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names}
    if -1 in object_ids:
        raise ValueError(f'Object body not found: {names}')
    groups = {'hand_visual': [], 'hand_collision': [], 'object_visual': [], 'object_collision': []}
    for geom in range(model.ngeom):
        is_object = model.geom_bodyid[geom] in object_ids
        collision = bool(model.geom_contype[geom] or model.geom_conaffinity[geom])
        label = ('object' if is_object else 'hand') + ('_collision' if collision else '_visual')
        groups[label].append(geom)
        model.geom_group[geom] = {'hand_visual': 1, 'object_visual': 2,
                                  'hand_collision': 3, 'object_collision': 4}[label]
        if not is_object:
            model.geom_rgba[geom] = [.25, .6, .95, 1.] if collision else [.6, .75, .85, .3]
    for i, geom in enumerate(groups['object_collision']):
        model.geom_rgba[geom] = [*colorsys.hsv_to_rgb(.04+.3*((i*.618034)%1), .65, .85), 1.]
    for geom in groups['object_visual']:
        model.geom_rgba[geom] = [.35, .8, .25, .3]
    return groups


def set_mesh_visibility(option, hand='collision', object_mesh='collision', contacts=False):
    if hand not in MESH_MODES or object_mesh not in MESH_MODES:
        raise ValueError('Invalid mesh display mode')
    option.geomgroup[:] = 0
    option.geomgroup[1] = hand in ('visual', 'both')
    option.geomgroup[2] = object_mesh in ('visual', 'both')
    option.geomgroup[3] = hand in ('collision', 'both')
    option.geomgroup[4] = object_mesh in ('collision', 'both')
    # A mesh geom is otherwise drawn as its detailed asset, even though collision
    # detection uses a convex hull. This flag is essential for collision inspection.
    option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contacts
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contacts


def mesh_counts(groups):
    return {label: len(geoms) for label, geoms in groups.items()}
