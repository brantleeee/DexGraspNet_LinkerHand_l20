"""Small, explicitly derived references; no overwrite/projection of source grasps."""
import copy
from pathlib import Path
import numpy as np
from .control import ResidualTargets
from .reference import digest


def variant_targets(reference,variant):
    if variant['parent_scene_sha256']!=reference['scene_sha256'] or variant['parent_targets']!=reference['targets']:
        raise ValueError('Variant parent identity differs')
    joints=reference['joints'];names=[j['name'] for j in joints]
    controller=ResidualTargets(joints,names)
    base=np.array([reference['targets'][n] for n in names],float);q=base.copy()
    for name,delta in variant.get('joint_deltas_rad',{}).items():
        if name not in names or names.index(name) not in controller.independent:raise ValueError('Only independent joints may be refined')
        if not np.isfinite(delta) or abs(delta)>.03:raise ValueError('Joint refinement exceeds 0.03 rad bound')
        q[names.index(name)]+=delta
    if np.any(q[controller.independent]<controller.lower[controller.independent]-1e-7) or np.any(q[controller.independent]>controller.upper[controller.independent]+1e-7):
        raise ValueError('Refinement violates joint/mimic-feasible limits')
    for c,p,m,b in controller.mimics:q[c]=m*q[p]+b
    if np.any(q<controller.lower-1e-7) or np.any(q>controller.upper+1e-7):raise ValueError('Refined mimic joint outside limits')
    translation=np.asarray(variant.get('wrist_translation_m',[0.,0.,0.]),float)
    rotation=np.asarray(variant.get('wrist_rotation_vector_deg',[0.,0.,0.]),float)
    if translation.shape!=(3,) or rotation.shape!=(3,) or not np.isfinite(translation).all() or not np.isfinite(rotation).all():raise ValueError('Invalid wrist transform')
    if np.linalg.norm(translation)>.002+1e-12 or np.linalg.norm(rotation)>2.+1e-12:raise ValueError('Wrist refinement exceeds 2mm/2degree bound')
    return dict(zip(names,map(float,q)))


def apply_variant(reference,variant,output):
    """Called inside SimulationApp; transform every hand rigid body and world anchor."""
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    from .environment import qmul
    targets=variant_targets(reference,variant);ref=copy.deepcopy(reference)
    ref.update(targets=targets,reference_is_successful=False,derived_variant=variant,parent_scene_sha256=reference['scene_sha256'])
    ref['qpos'].update(targets)
    d=np.asarray(variant.get('wrist_translation_m',[0.,0.,0.]),float)
    rv=np.radians(variant.get('wrist_rotation_vector_deg',[0.,0.,0.]));angle=np.linalg.norm(rv)
    if np.any(d) or angle:
        dq=np.r_[np.cos(angle/2),rv/max(angle,1e-30)*np.sin(angle/2)]
        R=np.array(Gf.Matrix3d().SetRotate(Gf.Quatd(float(dq[0]),Gf.Vec3d(*dq[1:])))).T
        stage=Usd.Stage.Open(reference['scene'])
        pivot=np.array(stage.GetPrimAtPath('/World/Hand/base_link').GetAttribute('xformOp:translate').Get())
        for prim in stage.GetPrimAtPath('/World/Hand').GetChildren():
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
            pos=prim.GetAttribute('xformOp:translate');orient=prim.GetAttribute('xformOp:orient')
            p=np.array(pos.Get());old=orient.Get();q=qmul(dq,np.r_[old.GetReal(),np.array(old.GetImaginary())])
            pos.Set(Gf.Vec3d(*(pivot+d+R@(p-pivot))));orient.Set(Gf.Quatf(float(q[0]),Gf.Vec3f(*q[1:])))
        anchor=UsdPhysics.FixedJoint.Get(stage,'/World/Hand/world_fixed');old=anchor.GetLocalRot0Attr().Get()
        q=qmul(dq,np.r_[old.GetReal(),np.array(old.GetImaginary())])
        anchor.GetLocalPos0Attr().Set(Gf.Vec3f(*(pivot+d)));anchor.GetLocalRot0Attr().Set(Gf.Quatf(float(q[0]),Gf.Vec3f(*q[1:])))
        path=Path(output)/'refined_scene.usda';stage.GetRootLayer().Export(str(path))
        ref['scene']=str(path);ref['scene_sha256']=digest(path)
        from scipy.spatial.transform import Rotation
        wrist=stage.GetPrimAtPath('/World/Hand/base_link');orientation=wrist.GetAttribute('xformOp:orient').Get()
        xyz=Rotation.from_quat([*orientation.GetImaginary(),orientation.GetReal()]).as_euler('xyz')
        for axis,value in zip('xyz',pivot+d):ref['qpos']['WRJT'+axis]=float(value)
        for axis,value in zip('xyz',xyz):ref['qpos']['WRJR'+axis]=float(value)
    return ref


def proposals(reference,joint_names,delta=.015,include_wrist=False):
    if not 0<delta<=.03:raise ValueError('Use a joint delta in (0,0.03] rad')
    base=dict(parent_scene_sha256=reference['scene_sha256'],parent_targets=reference['targets'])
    results=[]
    for name in joint_names:
        for sign in [-1,1]:
            v=dict(base,name=f'{name}_{"plus" if sign>0 else "minus"}',joint_deltas_rad={name:sign*delta})
            try:variant_targets(reference,v)
            except ValueError:continue
            results.append(v)
    if include_wrist:
        for axis in range(3):
            for sign in [-1,1]:
                v=dict(base,name=f'wrist_translate_{axis}_{sign}',joint_deltas_rad={},wrist_translation_m=[0.,0.,0.])
                v['wrist_translation_m'][axis]=sign*.001;results.append(v)
                v=dict(base,name=f'wrist_rotate_{axis}_{sign}',joint_deltas_rad={},wrist_rotation_vector_deg=[0.,0.,0.])
                v['wrist_rotation_vector_deg'][axis]=sign;results.append(v)
    return results
