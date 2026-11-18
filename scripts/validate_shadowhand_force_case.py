"""Independent ShadowHand drive-strength diagnostic fork with SDF collisions on every mesh.

Run using Isaac Sim's python.sh, after prepare_l20_isaacsim.py. No support plane,
object weld, convex hull or convex decomposition is used by this validator.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

GRAVITY = {'gx+': [9.81,0,0], 'gx-': [-9.81,0,0], 'gy+': [0,9.81,0],
           'gy-': [0,-9.81,0], 'gz+': [0,0,9.81], 'gz-': [0,0,-9.81]}


def rotation(rpy):
    x,y,z=rpy
    cx,cy,cz=np.cos([x,y,z]); sx,sy,sz=np.sin([x,y,z])
    return np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],
                     [sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]])


def transform(xyz, r):
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=xyz;return t


def axis_rotation(axis, angle):
    axis=np.array(axis,dtype=float);axis/=np.linalg.norm(axis)
    x,y,z=axis; k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)


def pregrasp_targets(joints,targets,fraction):
    """Open flexion joints toward neutral, preserving wrist and finger opposition."""
    opened=dict(targets)
    for joint in joints:
        name=joint['name']
        if joint['type']=='fixed' or joint['mimic']:continue
        if name.endswith(('_mcp_pitch','_pip','_cmc_pitch')) or name=='thumb_mcp':
            bounds=joint['limit'];neutral=float(np.clip(0.,bounds['lower'],bounds['upper']))
            opened[name]=(1-fraction)*targets[name]+fraction*neutral
    for joint in joints:
        if joint['mimic']:
            mimic=joint['mimic'];opened[joint['name']]=float(mimic.get('multiplier',1))*opened[mimic['joint']]+float(mimic.get('offset',0))
    for joint in joints:
        if joint['type']!='fixed' and not joint['limit']['lower']-1e-6<=opened[joint['name']]<=joint['limit']['upper']+1e-6:
            raise ValueError(f'Pregrasp outside URDF range: {joint["name"]}')
    return opened


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+')
    p.add_argument('--cases',choices=list(GRAVITY),nargs='+',default=list(GRAVITY))
    p.add_argument('--steps',type=int,default=2400,help='Gravity-on validation steps; default 2400 = 10 seconds at dt=1/240')
    p.add_argument('--protocol',choices=['acquire','direct'],default='acquire',help='acquire: open -> zero-gravity close/settle -> gravity; direct: historical closed-pose test')
    p.add_argument('--close-seconds',type=float,default=2.)
    p.add_argument('--settle-seconds',type=float,default=1.)
    p.add_argument('--open-fraction',type=float,default=.6)
    p.add_argument('--dt',type=float,default=1/240)
    p.add_argument('--sdf-resolution',type=int,default=256)
    p.add_argument('--friction',type=float,default=3.)
    p.add_argument('--effort-limit',type=float,default=1.,help='Independent joint actuator cap in Nm')
    p.add_argument('--kp',type=float,default=5.)
    p.add_argument('--kd',type=float,default=.05)
    p.add_argument('--max-slip',type=float,default=.005)
    p.add_argument('--max-angle-deg',type=float,default=15.)
    p.add_argument('--max-penetration',type=float,default=.001)
    p.add_argument('--gui',action='store_true')
    p.add_argument('--snapshots',action='store_true',help='Save initial/final viewport PNGs with the visible collision meshes')
    p.add_argument('--viewer-seconds',type=float,default=0.,help='GUI final hold; negative waits until window closes')
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    if not np.isfinite([a.kp,a.kd,a.effort_limit]).all() or a.kp<=0 or a.kd<0 or a.effort_limit<=0:p.error('Invalid drive parameters')
    if a.steps<=0 or a.dt<=0 or a.sdf_resolution<32:p.error('Invalid steps/dt/resolution')
    if a.close_seconds<=0 or a.settle_seconds<0 or not 0<a.open_fraction<=1:p.error('Invalid acquisition parameters')
    m=json.loads((a.inputs/'manifest.json').read_text())
    candidates=[c for c in m['candidates'] if a.indices is None or c['index'] in a.indices]
    if not candidates:p.error('No matching prepared candidates')
    if len(candidates)!=1:p.error('Select one candidate per process with --indices')
    a.output.mkdir(parents=True)
    def save(name,data): (a.output/name).write_text(json.dumps(data,indent=2))
    save('status.json',dict(status='starting',parameters=vars_serializable(a)))
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':not a.gui,'multi_gpu':False,'sync_loads':False,
                       'create_new_stage':False,'limit_cpu_threads':8,
                       'anti_aliasing':2,'denoiser':False})
    try:
        import omni.usd
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, UsdShade, UsdLux
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.core.utils.viewports import set_camera_view
        omni.usd.get_context().new_stage()
        stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage,1.)
        UsdGeom.Xform.Define(stage,'/World')
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        sim=SimulationContext(physics_dt=a.dt,rendering_dt=1/60,stage_units_in_meters=1.,backend='numpy')
        # Dynamic nonconvex SDF pairs require GPU dynamics. CPU tensor readback remains enabled.
        scene=UsdPhysics.Scene.Get(stage,'/physicsScene')
        scene.CreateGravityMagnitudeAttr(0.)
        sceneapi=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        sceneapi.CreateEnableGPUDynamicsAttr(True)
        sceneapi.CreateBroadphaseTypeAttr('GPU')
        sceneapi.CreateSolverTypeAttr('TGS')
        sceneapi.CreateEnableCCDAttr(True)
        sceneapi.CreateEnableStabilizationAttr(True)
        sceneapi.CreateTimeStepsPerSecondAttr(1/a.dt)
        material=UsdShade.Material.Define(stage,'/World/Material')
        mat=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(a.friction);mat.CreateDynamicFrictionAttr(a.friction);mat.CreateRestitutionAttr(0.)
        light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(1000.)
        meshes=np.load(a.inputs/'meshes.npz')
        audits=[]

        def quat(r):
            return Gf.Quatf(Gf.Matrix3d(*r.T.flatten().tolist()).ExtractRotation().GetQuat())

        def pose(xform,t):
            xform.AddTranslateOp().Set(Gf.Vec3d(*t[:3,3]))
            xform.AddOrientOp().Set(quat(t[:3,:3]))

        def inertia(prim,mass,com,matrix):
            d,v=np.linalg.eigh(matrix)
            if np.min(d)<=0:raise ValueError('Nonpositive inertia')
            if np.linalg.det(v)<0:v[:,0]*=-1
            api=UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(mass);api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*d));api.CreatePrincipalAxesAttr(quat(v))

        def sdf_mesh(path,key,xyz,rpy,color):
            mesh=UsdGeom.Mesh.Define(stage,path)
            vertices=meshes[key+'_vertices'];faces=meshes[key+'_faces']
            mesh.CreatePointsAttr(vertices.tolist())
            mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            pose(mesh,transform(xyz,rotation(rpy)))
            prim=mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf')
            sdf=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            sdf.CreateSdfResolutionAttr(a.sdf_resolution)
            sdf.CreateSdfSubgridResolutionAttr(6)
            collider=PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collider.CreateContactOffsetAttr(.0005);collider.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
            audits.append(dict(path=path,key=key,approximation='sdf',resolution=a.sdf_resolution,
                               grid_spacing_m=float(np.ptp(vertices,axis=0).max()/a.sdf_resolution),faces=len(faces)))

        # Build from URDF directly so no importer convex cooking can survive on a link.
        c=candidates[0];q=c['qpos']
        initial_targets=pregrasp_targets(m['joints'],c['targets'],a.open_fraction) if a.protocol=='acquire' else c['targets']
        close_steps=max(1,round(a.close_seconds/a.dt)) if a.protocol=='acquire' else 0
        settle_steps=round(a.settle_seconds/a.dt) if a.protocol=='acquire' else 0
        gravity_start=close_steps+settle_steps
        save('protocol.json',dict(protocol=a.protocol,initial_targets=initial_targets,closed_targets=c['targets'],
                                 close_steps=close_steps,settle_steps=settle_steps,gravity_steps=a.steps,
                                 gravity_start_index=gravity_start,object_free_during_acquisition=True,
                                 reset_object_at_gravity_switch=False))
        wrist=transform([q['WRJT'+v] for v in 'xyz'],rotation([q['WRJR'+v] for v in 'xyz']))
        fk={'base_link':wrist}
        pending=list(m['joints'])
        while pending:
            before=len(pending)
            for j in pending[:]:
                if j['parent'] not in fk:continue
                fk[j['child']]=fk[j['parent']]@transform(j['xyz'],rotation(j['rpy']))@transform([0,0,0],axis_rotation(j['axis'],initial_targets.get(j['name'],0.)))
                pending.remove(j)
            if len(pending)==before:raise ValueError('URDF graph is not a connected tree')
        UsdGeom.Xform.Define(stage,'/World/Hand')
        for link in m['links']:
            path='/World/Hand/'+link['name']
            body=UsdGeom.Xform.Define(stage,path);pose(body,fk[link['name']])
            prim=body.GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim)
            rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);rb.CreateDisableGravityAttr(False);rb.CreateEnableCCDAttr(True)
            it=link['inertia'];matrix=np.array([[it['ixx'],it['ixy'],it['ixz']],[it['ixy'],it['iyy'],it['iyz']],[it['ixz'],it['iyz'],it['izz']]])
            r=rotation(link['inertia_rpy']);inertia(prim,link['mass'],link['com'],r@matrix@r.T)
            for k,coll in enumerate(link['colliders']):sdf_mesh(path+f'/collision_{k}',coll['key'],coll['xyz'],coll['rpy'],(.25,.55,.85))
        base=stage.GetPrimAtPath('/World/Hand/base_link')
        UsdPhysics.ArticulationRootAPI.Apply(base)
        artapi=PhysxSchema.PhysxArticulationAPI.Apply(base)
        artapi.CreateEnabledSelfCollisionsAttr(False)
        artapi.CreateSolverPositionIterationCountAttr(32);artapi.CreateSolverVelocityIterationCountAttr(4)
        fix=UsdPhysics.FixedJoint.Define(stage,'/World/Hand/world_fixed')
        fix.CreateBody1Rel().SetTargets(['/World/Hand/base_link'])
        fix.CreateLocalPos0Attr(Gf.Vec3f(*wrist[:3,3]));fix.CreateLocalRot0Attr(quat(wrist[:3,:3]))
        drives={}
        for j in m['joints']:
            path='/World/Hand/joints/'+j['name']
            joint=(UsdPhysics.FixedJoint if j['type']=='fixed' else UsdPhysics.RevoluteJoint).Define(stage,path)
            joint.CreateBody0Rel().SetTargets(['/World/Hand/'+j['parent']]);joint.CreateBody1Rel().SetTargets(['/World/Hand/'+j['child']])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*j['xyz']))
            # Rotate the USD Z axis into the URDF joint axis in each link frame.
            axis=np.array(j['axis'],float);axis/=np.linalg.norm(axis)
            axisq=Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*axis)).GetQuat()
            ar=np.array(Gf.Matrix3d().SetRotate(axisq)).T
            joint.CreateLocalRot0Attr(quat(rotation(j['rpy'])@ar));joint.CreateLocalRot1Attr(Gf.Quatf(axisq))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
            if j['type']=='fixed':continue
            joint.CreateAxisAttr('Z')
            joint.CreateLowerLimitAttr(math.degrees(j['limit']['lower']));joint.CreateUpperLimitAttr(math.degrees(j['limit']['upper']))
            state=PhysxSchema.JointStateAPI.Apply(joint.GetPrim(),'angular');state.CreatePositionAttr(math.degrees(initial_targets[j['name']]))
            if j['mimic']:
                mimic=PhysxSchema.PhysxMimicJointAPI.Apply(joint.GetPrim(),'rotZ')
                mimic.CreateReferenceJointRel().SetTargets(['/World/Hand/joints/'+j['mimic']['joint']])
                mimic.CreateReferenceJointAxisAttr('rotZ')
                mimic.CreateGearingAttr(-float(j['mimic'].get('multiplier',1.)))
                mimic.CreateOffsetAttr(-math.degrees(float(j['mimic'].get('offset',0.))))
            else:
                drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular')
                # USD angular gains are per degree; PhysX tensor gains are per radian.
                drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(a.kp*math.pi/180);drive.CreateDampingAttr(a.kd*math.pi/180)
                drive.CreateMaxForceAttr(a.effort_limit);drive.CreateTargetPositionAttr(math.degrees(initial_targets[j['name']]))
                drives[j['name']]=drive
        body=UsdGeom.Xform.Define(stage,'/World/Object');pose(body,np.eye(4))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateEnableCCDAttr(True)
        inertia(body.GetPrim(),m['object_mass_kg'],m['object_com'],np.array(m['object_inertia']))
        sdf_mesh('/World/Object/collision','object',[0,0,0],[0,0,0],(.9,.45,.12))
        # Audit all active colliders; any residual convex approximation aborts the run.
        active=[]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                approx=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                if approx!='sdf' or not prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):raise RuntimeError(f'Non-SDF collider: {prim.GetPath()} {approx}')
                active.append(str(prim.GetPath()))
        save('collision_audit.json',dict(colliders=audits,active_count=len(active),all_sdf=True,object_mesh_kind=m['object_mesh_kind'],source=m['meshes']))
        away=-wrist[:3,3].copy();away[2]=0.
        if np.linalg.norm(away)<1e-6:away=np.array([1.,1.,0.])
        away/=np.linalg.norm(away)
        set_camera_view(eye=.4*away+np.array([0.,0.,.22]),target=np.array([0.,0.,.02]))
        stage.GetRootLayer().Export(str((a.output/'scene.usda').resolve()))
        save('status.json',dict(status='cooking_sdf',colliders=len(active)))
        hand=Articulation('/World/Hand/base_link',reset_xform_properties=False)
        obj=RigidPrim('/World/Object',reset_xform_properties=False,track_contact_forces=True,
                      contact_filter_prim_paths_expr=['/World/Hand/'+l['name'] for l in m['links']],max_contact_count=4096)
        sim.initialize_physics();sim.play()
        hand.initialize()
        obj.initialize()
        names=list(hand.dof_names)
        if set(names)!={j['name'] for j in m['joints'] if j['type']!='fixed'}:raise RuntimeError(f'DOF mismatch: {names}')
        gains=hand.get_gains()
        independent=[names.index(n) for n in drives]
        if not np.allclose(np.asarray(gains[0])[0,independent],a.kp,rtol=1e-4):raise RuntimeError('Angular stiffness unit conversion failed')
        if not np.allclose(np.asarray(gains[1])[0,independent],a.kd,rtol=1e-4):raise RuntimeError('Angular damping unit conversion failed')
        save('runtime_drives.json',dict(dof_names=names,kp=np.asarray(gains[0]).tolist(),kd=np.asarray(gains[1]).tolist(),
                                       joint_velocity_limit_enforced=False,hand_gravity_enabled=True))
        caps=np.asarray(hand.get_max_efforts())
        if not np.allclose(caps[0,independent],a.effort_limit,rtol=1e-4):raise RuntimeError('Actuator cap readback differs')
        save('drive_limits.json',dict(effort_limit_nm=a.effort_limit,readback=caps.tolist(),base_validator_sha256='cef03f52f112bd5a7228e21f917d3fc38ac9c85cc3186ca4a0f69661d57be665',measured_effort_semantics='solver joint effort in DOF direction; not isolated motor torque',pd_demand_semantics='post-step kp*(target-q)-kd*qdot estimate; not exact implicit solver drive torque'))
        def snapshot(name):
            if not a.snapshots:return
            import asyncio
            from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
            sim.pause()
            for _ in range(8):sim.render()
            capture=capture_viewport_to_file(get_active_viewport(),file_path=str((a.output/name).resolve()))
            task=asyncio.ensure_future(capture.wait_for_result())
            for _ in range(120):
                sim.render()
                if task.done():break
            if not task.done():raise RuntimeError('Viewport snapshot did not complete')
            task.result()
            sim.play()
        results=[]
        # Fixed root is authored per candidate; launch separate process for different candidates.
        for case in a.cases:
            targets=np.array([[c['targets'][n] for n in names]],dtype=np.float32)
            opened=np.array([[initial_targets[n] for n in names]],dtype=np.float32)
            # Independent cases restart only here; gravity switching below never teleports or arrests the object.
            scene.CreateGravityMagnitudeAttr(0.)
            hand.set_joint_positions(opened);hand.set_joint_velocities(np.zeros_like(targets))
            hand.set_joint_position_targets(opened)
            obj.set_world_poses(np.zeros((1,3),np.float32),np.array([[1,0,0,0]],np.float32));obj.set_velocities(np.zeros((1,6),np.float32))
            g=np.array(GRAVITY[case],float);scene.CreateGravityDirectionAttr(Gf.Vec3f(*(g/np.linalg.norm(g))))
            positions=[];orientations=[];joints=[];velocities=[];forces=[];depths=[];contacts=[];phases=[];gravities=[];commands=[]
            initial_joint_error=float(np.max(np.abs(hand.get_joint_positions()-opened)))
            reference_pos=np.zeros(3);reference_quat=np.array([1.,0,0,0]);reference_velocity=np.zeros(6)
            joint_velocity_trace=[];pd_demand_trace=[];solver_effort_trace=[]
            acquired=a.protocol=='direct';pregravity_contact_count=0;pregravity_force=0.
            if case==a.cases[0]:snapshot('initial.png')
            start=time.monotonic()
            total_steps=gravity_start+a.steps
            for step in range(total_steps):
                if step<close_steps:
                    phase=0;u=(step+1)/close_steps;blend=10*u**3-15*u**4+6*u**5
                    command=opened+(targets-opened)*blend
                elif step<gravity_start:
                    phase=1;command=targets
                else:
                    phase=2;command=targets
                if step==gravity_start:
                    if a.protocol=='acquire':
                        pos0,quat0=obj.get_world_poses();reference_pos=pos0[0].copy();reference_quat=quat0[0].copy()
                        reference_velocity=obj.get_velocities()[0].copy()
                        pregravity_contact_count=contacts[-1];pregravity_force=forces[-1]
                        acquired=pregravity_contact_count>0
                    # No object pose/velocity reset at this boundary.
                    scene.CreateGravityMagnitudeAttr(float(np.linalg.norm(g)))
                hand.set_joint_position_targets(command)
                # One iteration must be exactly one physics step. step(render=True)
                # can advance multiple substeps when rendering_dt > physics_dt.
                sim.step(render=False)
                if a.gui and (step % max(1,round((1/60)/a.dt))==0 or step==a.steps-1):
                    sim.render()
                if step==0:
                    from omni.physx import get_physx_statistics_interface
                    from omni.physx.bindings._physx import PhysicsSceneStats
                    from pxr import UsdUtils, PhysicsSchemaTools
                    stats=PhysicsSceneStats()
                    ok=get_physx_statistics_interface().get_physx_scene_statistics(
                        UsdUtils.StageCache.Get().Insert(stage).ToLongInt(),
                        PhysicsSchemaTools.sdfPathToInt('/physicsScene'),stats)
                    fields={k:getattr(stats,k) for k in dir(stats) if k.startswith('nb_') and isinstance(getattr(stats,k),(int,float))}
                    save('runtime_collision_statistics.json',dict(available=ok,statistics=fields))
                    if not ok:raise RuntimeError('Cannot audit runtime collision statistics')
                    if fields.get('nb_convex_shapes',0)>0:raise RuntimeError('Convex shapes detected at runtime')
                    if fields.get('nb_trimesh_shapes')!=len(active):raise RuntimeError('Some SDF mesh shapes did not load')
                pos,ori=obj.get_world_poses();jpos=hand.get_joint_positions();vel=obj.get_velocities()
                jvel=hand.get_joint_velocities();solver_effort=hand.get_measured_joint_efforts()
                joint_velocity_trace.append(jvel[0].copy());pd_demand_trace.append((a.kp*(command-jpos)-a.kd*jvel)[0].copy());solver_effort_trace.append(solver_effort[0].copy())
                normal,points,normals,separations,counts,starts=obj.get_contact_force_data(dt=a.dt)
                count=int(np.sum(counts))
                used=np.concatenate([np.arange(int(s),int(s)+int(n)) for s,n in zip(starts.ravel(),counts.ravel())]) if count else np.array([],dtype=int)
                maxdepth=float(max(0.,-np.min(separations[used]))) if count else 0.
                force=float(np.linalg.norm(obj.get_contact_force_matrix(dt=a.dt),axis=-1).sum())
                if not all(np.isfinite(x).all() for x in (pos,ori,jpos,vel)):
                    save('numerical_failure.json',dict(candidate=c['index'],case=case,
                         phase=['close_zero_g','settle_zero_g','gravity_test'][phase],
                         step=step+1,total_steps=total_steps,
                         invalid_fields=[name for name,value in [('object_position',pos),('object_orientation',ori),
                                         ('joint_positions',jpos),('object_velocity',vel)] if not np.isfinite(value).all()]))
                    raise RuntimeError(f'Nonfinite physics state: candidate={c["index"]}, case={case}, step={step+1}, phase={phase}')
                positions.append(pos[0].copy());orientations.append(ori[0].copy());joints.append(jpos[0].copy());velocities.append(vel[0].copy());forces.append(force);depths.append(maxdepth);contacts.append(count)
                phases.append(phase);gravities.append(g.copy() if phase==2 else np.zeros(3));commands.append(command[0].copy())
                if step%100==0 or step==gravity_start:save('status.json',dict(status='simulating',case=case,phase=['close_zero_g','settle_zero_g','gravity_test'][phase],step=step+1,total=total_steps))
                if a.gui:
                    remaining=(step+1)*a.dt-(time.monotonic()-start)
                    if remaining>0:time.sleep(remaining)
            positions=np.array(positions);orientations=np.array(orientations)
            gravity_positions=positions[gravity_start:];gravity_orientations=orientations[gravity_start:]
            slip=np.linalg.norm(gravity_positions-reference_pos,axis=1)
            angles=np.degrees(2*np.arccos(np.clip(np.abs(gravity_orientations@reference_quat),0,1)))
            tail=max(1,int(1/a.dt));hold=np.array(forces[-tail:])>.001
            numerical=bool(np.isfinite(joints).all() and np.isfinite(positions).all() and np.isfinite(orientations).all())
            gravity_passed=bool(numerical and slip.max()<=a.max_slip and angles.max()<=a.max_angle_deg and max(depths)<=a.max_penetration and len(hold)>=tail and hold.all())
            result=dict(candidate=c['index'],case=case,protocol=a.protocol,steps=a.steps,seconds=a.steps*a.dt,
                        total_steps=total_steps,total_seconds=total_steps*a.dt,gravity_start_index=gravity_start,
                        acquired_before_gravity=acquired,pregravity_contact_count=pregravity_contact_count,pregravity_contact_force_n=pregravity_force,
                        acquisition_displacement_m=float(np.linalg.norm(reference_pos)),gravity_reference_position=reference_pos.tolist(),
                        gravity_reference_quaternion=reference_quat.tolist(),gravity_switch_velocity=reference_velocity.tolist(),
                        max_joint_closure_rad=float(np.max(np.abs(np.asarray(joints)[max(gravity_start-1,0)]-opened[0]))),
                        gravity_only_passed=gravity_passed,
                        max_slip_m=float(slip.max()),final_slip_m=float(slip[-1]),max_angle_deg=float(angles.max()),
                        max_penetration_m=float(max(depths)),final_contact_force_n=forces[-1],
                        continuous_last_second_contact=bool(len(hold)>=tail and hold.all()),numerical_valid=numerical,
                        initial_joint_error_rad=initial_joint_error,
                        drive_effort_limit_nm=a.effort_limit,max_joint_target_error_rad=float(np.max(np.abs(np.asarray(joints)-np.asarray(commands)))),max_estimated_pd_demand_nm=float(np.max(np.abs(pd_demand_trace))),estimated_pd_demand_over_cap_fraction=float(np.mean(np.abs(pd_demand_trace)>=a.effort_limit)),
                        passed=bool(acquired and gravity_passed))
            np.savez_compressed(a.output/f'trajectory_{case}.npz',time=np.arange(1,total_steps+1)*a.dt,
                                object_position=positions,object_quaternion=orientations,joint_positions=joints,
                                joint_velocities=joint_velocity_trace,estimated_pd_demand=pd_demand_trace,solver_joint_efforts=solver_effort_trace,object_velocity=velocities,contact_force=forces,penetration=depths,dof_names=names,
                                phase=np.asarray(phases,dtype=np.int8),gravity=gravities,joint_targets=commands,contact_count=contacts,
                                initial_joint_positions=opened[0],initial_object_position=np.zeros(3),initial_object_quaternion=[1.,0,0,0],
                                gravity_reference_position=reference_pos,gravity_reference_quaternion=reference_quat,gravity_start_index=gravity_start)
            results.append(result);save('results.json',results)
            snapshot(f'final_{case}.png')
        summary=dict(parameters=vars_serializable(a),manifest=m,collision_audit='collision_audit.json',results=results,
                     all_directions_pass=bool(len(results)==6 and all(r['passed'] for r in results)))
        save('summary.json',summary)
        (a.output/'REPORT.md').write_text('# ShadowHand 加力诊断 Isaac Sim SDF 验证\n\n'
            +f"物体：{m['object']}，候选：{c['index']}，scale={m['scale']}，质量={m['object_mass_kg']:.6f} kg。\n\n"
            +f"协议：{a.protocol}；零重力闭合 {close_steps*a.dt:.3f} 秒、零重力保持 {settle_steps*a.dt:.3f} 秒、重力验证 {a.steps*a.dt:.3f} 秒。开启重力时不重置物体位姿或速度。\n\n"
            +'碰撞：全部手部 STL 与物体网格均使用 SDF；无凸包、无物体支撑或固定约束。\n\n'
            +f"物体来源：{m['object_mesh_kind']}。旧 CoACD 来源不等于原始高精度物体。\n\n"
            +'通过标准：开启重力前存在接触；重力阶段相对开启前的位姿偏移≤5 mm、旋转≤15°；包括闭合在内的全部采样步接触穿透≤1 mm；末尾连续1秒有接触，且数值有效。direct 历史模式不要求获取阶段。\n\n'
            +'|方向|开重力前接触数|闭合推移 mm|重力阶段最大位移 mm|最大转角 °|全程最大穿透 mm|通过|\n|---|---:|---:|---:|---:|---:|---|\n'
            +''.join(f"|{r['case']}|{r['pregravity_contact_count']}|{r['acquisition_displacement_m']*1000:.2f}|{r['max_slip_m']*1000:.2f}|{r['max_angle_deg']:.2f}|{r['max_penetration_m']*1000:.2f}|{r['passed']}|\n" for r in results))
        save('status.json',dict(status='complete',all_directions_pass=summary['all_directions_pass']))
        if a.gui:
            sim.pause()
            until=time.monotonic()+a.viewer_seconds
            while app.is_running() and (a.viewer_seconds<0 or time.monotonic()<until):app.update()
        sim.stop()
    except BaseException as exc:
        import traceback
        save('error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        app.close()


def vars_serializable(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


if __name__=='__main__':main()
