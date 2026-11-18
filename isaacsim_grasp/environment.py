"""Synchronized vector environments on the original SDF USD + physical mimic model.

Imported only after SimulationApp is started. Episodes share phase timing so true
zero gravity can be used, without object-support forces or per-body compensation.
"""
import math
from pathlib import Path
import numpy as np
from .control import ResidualTargets

CASES=['gx+','gx-','gy+','gy-','gz+','gz-']


def qmul(a,b):
    a=np.asarray(a);b=np.asarray(b)
    w=a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdims=True)
    v=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])
    return np.concatenate([w,v],axis=-1)


def qconj(q):
    out=q.copy();out[...,1:]*=-1;return out


def rotation(axis,angle):
    from pxr import Gf
    quat=Gf.Rotation(Gf.Vec3d(*axis),angle).GetQuat()
    return np.array(Gf.Matrix3d().SetRotate(quat)).T,np.array([quat.GetReal(),*quat.GetImaginary()])


class ResidualGraspEnv:
    def __init__(self,reference,config,app):
        import omni.usd
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import Articulation,RigidPrim
        from isaacsim.core.cloner import Cloner
        self.ref=reference;self.cfg=config;self.app=app;self.n=config['num_envs'];self.dt=1/240;self.decimation=4
        if self.n%6:raise ValueError('num-envs must be a multiple of six')
        self.close_steps=round(config['close_seconds']/self.dt);self.settle_steps=round(config['settle_seconds']/self.dt)
        self.gravity_start=self.close_steps+self.settle_steps;self.gravity_steps=round(config['gravity_seconds']/self.dt)
        self.total_steps=self.gravity_start+self.gravity_steps
        if self.total_steps%self.decimation:raise ValueError('Episode length must align with four physics substeps')
        self.rng=np.random.default_rng(config['seed']);self.case_ids=np.arange(self.n)%6
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage();self.stage=stage
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
        self.sim=SimulationContext(physics_dt=self.dt,rendering_dt=1/60,stage_units_in_meters=1.,backend='numpy')
        self.scene=UsdPhysics.Scene.Get(stage,'/physicsScene');self.scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));self.scene.CreateGravityMagnitudeAttr(0.)
        api=PhysxSchema.PhysxSceneAPI.Apply(self.scene.GetPrim());api.CreateEnableGPUDynamicsAttr(True);api.CreateBroadphaseTypeAttr('GPU')
        api.CreateSolverTypeAttr('TGS');api.CreateEnableCCDAttr(True);api.CreateEnableStabilizationAttr(True);api.CreateTimeStepsPerSecondAttr(240.)
        source=Path(reference['scene']);self.paths=[];self.rot=[];self.quat=[];self.origins=[]
        transforms=[([0,1,0],90),([0,1,0],-90),([1,0,0],-90),([1,0,0],90),([1,0,0],180),([1,0,0],0)]
        side=math.ceil(math.sqrt(self.n));UsdGeom.Xform.Define(stage,'/World/envs')
        for i in range(self.n):
            path=f'/World/envs/env_{i:04d}';self.paths.append(path)
            env=UsdGeom.Xform.Define(stage,path);env.GetPrim().GetReferences().AddReference(str(source),'/World')
            r,q=rotation(*transforms[self.case_ids[i]]);origin=np.array([(i%side)*1.0,(i//side)*1.0,0.])
            env.AddTranslateOp().Set(Gf.Vec3d(*origin));env.AddOrientOp().Set(Gf.Quatf(float(q[0]),Gf.Vec3f(*q[1:])))
            self.rot.append(r);self.quat.append(q);self.origins.append(origin)
            # Joint body0 is the world: its anchor must be transformed explicitly.
            joint=UsdPhysics.FixedJoint.Get(stage,path+'/Hand/world_fixed')
            pos=np.array(joint.GetLocalPos0Attr().Get());orient=joint.GetLocalRot0Attr().Get()
            oq=np.array([orient.GetReal(),*orient.GetImaginary()]);worldq=qmul(q,oq)
            joint.CreateLocalPos0Attr(Gf.Vec3f(*(r@pos+origin)))
            joint.CreateLocalRot0Attr(Gf.Quatf(float(worldq[0]),Gf.Vec3f(*worldq[1:])))
            # One shared light is enough; disable inherited lights in other copies.
            if i:stage.GetPrimAtPath(path+'/Light').SetActive(False)
        self.rot=np.asarray(self.rot);self.quat=np.asarray(self.quat);self.origins=np.asarray(self.origins)
        Cloner(stage=stage).filter_collisions('/physicsScene','/World/collisionGroups',self.paths,[])
        self.hand=Articulation('/World/envs/env_.*/Hand/base_link',reset_xform_properties=False)
        filters=[f'/World/envs/env_.*/Hand/{link}' for link in reference['hand_links']]
        self.obj=RigidPrim('/World/envs/env_.*/Object',reset_xform_properties=False,track_contact_forces=True,
            contact_filter_prim_paths_expr=filters,max_contact_count=4096*self.n)
        self.sim.initialize_physics();self.sim.play();self.hand.initialize();self.obj.initialize()
        if self.hand.count!=self.n or self.obj.count!=self.n:raise RuntimeError('Vector view count mismatch')
        if list(self.obj.prim_paths)!=[p+'/Object' for p in self.paths]:raise RuntimeError('Object view ordering changed')
        if list(self.hand.prim_paths)!=[p+'/Hand/base_link' for p in self.paths]:raise RuntimeError('Hand view ordering changed')
        self.names=list(self.hand.dof_names)
        self.controller=ResidualTargets(reference['joints'],self.names,config['residual_scale'],config['residual_rate'])
        if len(self.controller.independent)!=16:raise RuntimeError('Expected 16 independent controls')
        self.closed=np.array([reference['targets'][n] for n in self.names],np.float32)
        # Use the same pregrasp construction as the dynamics validator.
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from validate_l20_isaacsim import pregrasp_targets
        opened=pregrasp_targets(reference['joints'],reference['targets'],config['open_fraction'])
        self.opened=np.array([opened[n] for n in self.names],np.float32)
        kp,kd=self.hand.get_gains();ids=self.controller.independent
        if not np.allclose(kp[:,ids],5,rtol=1e-4) or not np.allclose(kd[:,ids],.05,rtol=1e-4):raise RuntimeError('Reference PD gains changed')
        active=[]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='sdf' or not prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):raise RuntimeError('Non-SDF collider')
                active.append(str(prim.GetPath()))
        if len(active)!=23*self.n:raise RuntimeError('Some collision shapes were not composed')
        for path in self.paths:
            for j in reference['joints']:
                if not j['mimic']:continue
                prim=stage.GetPrimAtPath(path+'/Hand/joints/'+j['name'])
                mimic=PhysxSchema.PhysxMimicJointAPI(prim,'rotZ')
                targets=mimic.GetReferenceJointRel().GetTargets()
                if list(map(str,targets))!=[path+'/Hand/joints/'+j['mimic']['joint']]:
                    raise RuntimeError('Cloned mimic constraint references another environment')
        self.audit=dict(active_sdf_colliders=len(active),environments=self.n,kp=np.asarray(kp).tolist(),kd=np.asarray(kd).tolist(),
            audited_physical_mimic_constraints=5*self.n,
            mimic='physical_PhysxMimicJointAPI',collision_groups='one_filtered_group_per_environment',joint_velocity_limit_enforced=False,
            gravity_phases='synchronized true zero gravity then world -Z; six initial scene rotations',
            residual_rate_is_target_offset_limit_not_actual_joint_velocity=True)
        self.reset()

    def local(self,v):return np.einsum('nji,nj->ni',self.rot,v)

    def reset(self):
        self.scene.CreateGravityMagnitudeAttr(0.);self.physics_step=0;self.controller.reset(self.n)
        q=np.tile(self.opened,(self.n,1));self.hand.set_joint_positions(q);self.hand.set_joint_velocities(np.zeros_like(q));self.hand.set_joint_position_targets(q)
        noise=self.rng.uniform(-self.cfg['position_noise'],self.cfg['position_noise'],(self.n,3))
        pos=self.origins+np.einsum('nij,nj->ni',self.rot,noise)
        self.obj.set_world_poses(pos.astype(np.float32),self.quat.astype(np.float32));self.obj.set_velocities(np.zeros((self.n,6),np.float32))
        self.previous_action=np.zeros((self.n,16),np.float32);self.last_target=q.copy();self.current_reference=q.copy()
        self.ref_pos=np.zeros((self.n,3));self.ref_quat=np.tile([1.,0,0,0],(self.n,1))
        self.max_slip=np.zeros(self.n);self.max_angle=np.zeros(self.n);self.max_depth=np.zeros(self.n)
        self.acquired=np.zeros(self.n,bool);self.contact_steps=np.zeros(self.n,int);self.dropped=np.zeros(self.n,bool)
        self.max_mimic_error=np.zeros(self.n)
        self.force=np.zeros(self.n);self.depth=np.zeros(self.n);self.count=np.zeros(self.n,int);self.runtime_checked=False
        return self.observe()

    def state(self):
        pos,quat=self.obj.get_world_poses();q=self.hand.get_joint_positions();qd=self.hand.get_joint_velocities();vel=self.obj.get_velocities()
        if not all(np.isfinite(v).all() for v in [pos,quat,q,qd,vel]):raise RuntimeError(f'Nonfinite RL physics at step {self.physics_step}; no silent reset')
        return self.local(pos-self.origins),qmul(qconj(self.quat),quat),q,qd,self.local(vel[:,:3]),self.local(vel[:,3:])

    def contact(self):
        normal,points,normals,separation,counts,starts=self.obj.get_contact_force_data(dt=self.dt)
        counts=np.asarray(counts).reshape(self.n,-1);starts=np.asarray(starts).reshape(self.n,-1)
        self.count=counts.sum(1).astype(int);self.depth=np.zeros(self.n)
        for i in range(self.n):
            parts=[np.arange(int(s),int(s)+int(c)) for s,c in zip(starts[i],counts[i]) if c]
            if parts:self.depth[i]=max(0.,-float(np.min(separation[np.concatenate(parts)])))
        self.force=np.linalg.norm(self.obj.get_contact_force_matrix(dt=self.dt),axis=-1).sum(1)
        if not np.isfinite(self.force).all() or not np.isfinite(self.depth).all():raise RuntimeError('Nonfinite contact data')

    def observe(self):
        pos,quat,q,qd,linear,angular=self.state()
        gravity=self.physics_step>=self.gravity_start
        refpos=self.ref_pos if gravity else np.zeros_like(pos);refquat=self.ref_quat if gravity else np.tile([1.,0,0,0],(self.n,1))
        error=qmul(qconj(refquat),quat);error*=np.where(error[:,:1]<0,-1.,1.)
        phase=np.zeros((self.n,3));phase[:,0 if self.physics_step<self.close_steps else 1 if not gravity else 2]=1
        obs=np.concatenate([(q-self.current_reference)/.2,qd/5,self.controller.offset/self.cfg['residual_scale'],
            (pos-refpos)/.02,error[:,1:],linear/.1,angular,self.local(np.tile([0.,0.,-1.],(self.n,1))),
            np.log1p(self.force[:,None])/5,phase,self.previous_action],axis=1)
        return np.clip(obs,-10,10).astype(np.float32)

    def audit_runtime(self):
        from omni.physx import get_physx_statistics_interface
        from omni.physx.bindings._physx import PhysicsSceneStats
        from pxr import UsdUtils,PhysicsSchemaTools
        stats=PhysicsSceneStats();ok=get_physx_statistics_interface().get_physx_scene_statistics(
            UsdUtils.StageCache.Get().Insert(self.stage).ToLongInt(),PhysicsSchemaTools.sdfPathToInt('/physicsScene'),stats)
        if not ok or stats.nb_convex_shapes!=0 or stats.nb_trimesh_shapes!=23*self.n:raise RuntimeError('RL runtime SDF collision audit failed')
        self.audit['runtime_convex_shapes']=stats.nb_convex_shapes;self.audit['runtime_trimesh_shapes']=stats.nb_trimesh_shapes
        self.runtime_checked=True

    def step(self,action):
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=(self.n,16) or not np.isfinite(action).all():raise ValueError('Invalid action')
        for _ in range(self.decimation):
            k=self.physics_step
            u=min(1.,(k+1)/self.close_steps);blend=10*u**3-15*u**4+6*u**5
            self.current_reference=np.tile(self.opened+(self.closed-self.opened)*blend,(self.n,1))
            self.last_target=self.controller.step(self.current_reference,action,self.dt,gate=blend)
            self.hand.set_joint_position_targets(self.last_target);self.sim.step(render=False)
            self.physics_step+=1
            if not self.runtime_checked:self.audit_runtime()
            self.contact();pos,quat,q,qd,linear,angular=self.state()
            if self.physics_step==self.gravity_start:
                self.ref_pos=pos.copy();self.ref_quat=quat.copy();self.acquired=self.count>0
                # The boundary observation must already expose the captured
                # reference; gravity acts on the following physics step.
                self.scene.CreateGravityMagnitudeAttr(9.81)
            for c,p,m,b in self.controller.mimics:
                self.max_mimic_error=np.maximum(self.max_mimic_error,np.abs(q[:,c]-m*q[:,p]-b))
            self.max_depth=np.maximum(self.max_depth,self.depth)
            if k>=self.gravity_start:
                slip=np.linalg.norm(pos-self.ref_pos,axis=1);angle=np.degrees(2*np.arccos(np.clip(np.abs((quat*self.ref_quat).sum(1)),0,1)))
                self.max_slip=np.maximum(self.max_slip,slip);self.max_angle=np.maximum(self.max_angle,angle)
                self.contact_steps=np.where(self.force>.001,self.contact_steps+1,0)
        gravity=self.physics_step>self.gravity_start
        if gravity:
            position=np.linalg.norm(pos-self.ref_pos,axis=1);angle=np.degrees(2*np.arccos(np.clip(np.abs((quat*self.ref_quat).sum(1)),0,1)))
        else:position=np.linalg.norm(pos,axis=1);angle=np.degrees(2*np.arccos(np.clip(np.abs(quat[:,0]),0,1)))
        speed=np.linalg.norm(linear,axis=1);spin=np.linalg.norm(angular,axis=1)
        new_drop=gravity & (~self.dropped) & ((position>.05)|(angle>90))
        self.dropped|=new_drop
        good=(position<=.005)&(angle<=15)&(self.depth<=.001)&(self.force>.001)&(speed<.02)&(spin<.3)
        pose=np.exp(-np.minimum((position/.01)**2+(angle/20)**2,40))
        # Positive rewards require a near-reference, low-motion object; a fallen
        # object cannot earn reward merely by becoming stationary far away.
        reward=(1.5*pose+.5*pose*(self.force>.001)+2.*good*gravity)*np.exp(-np.minimum((speed/.04)**2+(spin/.6)**2,40))
        reward-=.15*np.mean((self.controller.offset/self.cfg['residual_scale'])**2,axis=1)
        reward-=.08*np.mean((np.tanh(action)-np.tanh(self.previous_action))**2,axis=1)
        reward-=.03*np.mean(np.minimum(qd**2,25),axis=1)+2*np.minimum(self.depth/.001,5)
        reward=np.where(self.dropped,0,reward)-5*new_drop
        self.previous_action=action.copy();done=self.physics_step>=self.total_steps
        info={}
        if done:
            passed=self.acquired&(self.max_slip<=.005)&(self.max_angle<=15)&(self.max_depth<=.001)&(self.contact_steps>=240)
            info=dict(passed=passed.tolist(),cases=[CASES[c] for c in self.case_ids],max_slip_m=self.max_slip.tolist(),
                max_angle_deg=self.max_angle.tolist(),max_penetration_m=self.max_depth.tolist(),acquired=self.acquired.tolist(),
                max_mimic_error_rad=self.max_mimic_error.tolist(),
                dropped=self.dropped.tolist(),last_second_contact=(self.contact_steps>=240).tolist())
        return self.observe(),reward.astype(np.float32),done,info

    def render(self):
        self.sim.render()

    def close(self):self.sim.stop()
