"""Native Isaac Sim RGB export of saved states. No physics rerun or compositing."""
import argparse,json,sys,hashlib,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from validate_l20_isaacsim import rotation,axis_rotation,transform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--case',default='gy+')
    p.add_argument('--output',type=Path,required=True,help='New folder: raw PNG frames, MP4, provenance')
    p.add_argument('--fps',type=int,default=24);p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=720)
    p.add_argument('--start-seconds',type=float,default=0.);p.add_argument('--end-seconds',type=float,default=13.)
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    if a.fps<=0 or a.width<=0 or a.height<=0 or a.start_seconds<0 or a.end_seconds<a.start_seconds:p.error('Invalid capture parameters')
    result=json.loads((a.run/'result.json').read_text());ref=json.loads((a.run/'reference_snapshot.json').read_text())
    if not result['complete']:p.error('Run incomplete')
    i=result['info']['cases'].index(a.case)
    with np.load(a.run/'trajectory.npz') as z:d={k:z[k] for k in z.files}
    source=Path(ref['scene']);sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if sha(source)!=ref['scene_sha256']:p.error('Scene hash differs from reference')
    a.output.mkdir(parents=True);frames_dir=a.output/'frames';frames_dir.mkdir()
    def save(name,value):(a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    save('status.json',dict(status='starting'))
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,multi_gpu=False,sync_loads=False,create_new_stage=False,
        limit_cpu_threads=8,width=a.width,height=a.height,renderer='RaytracedLighting',anti_aliasing=2))
    try:
        import omni.usd,omni.timeline
        import omni.replicator.core as rep
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Sdf,Gf,UsdLux
        from PIL import Image
        context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
        stage.GetRootLayer().TransferContent(Sdf.Layer.FindOrOpen(str(source.resolve())))
        timeline=omni.timeline.get_timeline_interface();timeline.stop();timeline.set_auto_update(False)
        collider_count=0
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collider_count+=1;UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        for path in ['/physicsScene','/World/Hand/joints','/World/Hand/world_fixed']:
            prim=stage.GetPrimAtPath(path)
            if prim:prim.SetActive(False)
        if collider_count!=23:raise ValueError('Expected original 23 SDF input meshes')
        ops={}
        for name in ref['hand_links']:
            ops[name]={op.GetOpName():op for op in UsdGeom.Xformable(stage.GetPrimAtPath('/World/Hand/'+name)).GetOrderedXformOps()}
        object_ops={op.GetOpName():op for op in UsdGeom.Xformable(stage.GetPrimAtPath('/World/Object')).GetOrderedXformOps()}
        wrist=ref['qpos'];base=transform([wrist['WRJT'+k] for k in 'xyz'],rotation([wrist['WRJR'+k] for k in 'xyz']))
        ordered=[];known={'base_link'};pending=list(ref['joints'])
        while pending:
            ready=[j for j in pending if j['parent'] in known]
            if not ready:raise ValueError('Invalid joint tree')
            for j in ready:ordered.append(j);known.add(j['child']);pending.remove(j)
        origins={j['name']:transform(j['xyz'],rotation(j['rpy'])) for j in ordered}
        names={str(n.decode() if isinstance(n,bytes) else n):j for j,n in enumerate(d['dof_names'])}
        def apply(k):
            fk={'base_link':base}
            for j in ordered:
                angle=0. if j['type']=='fixed' else float(d['joint_position'][k,i,names[j['name']]])
                fk[j['child']]=fk[j['parent']]@origins[j['name']]@transform([0,0,0],axis_rotation(j['axis'],angle))
            for name,t in fk.items():
                ops[name]['xformOp:translate'].Set(Gf.Vec3d(*t[:3,3]))
                ops[name]['xformOp:orient'].Set(Gf.Quatf(Gf.Matrix3d(*t[:3,:3].T.flatten().tolist()).ExtractRotation().GetQuat()))
            pos=d['object_position'][k,i];q=d['object_quaternion'][k,i]
            object_ops['xformOp:translate'].Set(Gf.Vec3d(*pos))
            object_ops['xformOp:orient'].Set(Gf.Quatf(float(q[0]),Gf.Vec3f(*q[1:])))
        apply(0)
        camera=UsdGeom.Camera.Define(stage,'/World/RawVideoCamera')
        eye=Gf.Vec3d(.40,-.32,.30);target=Gf.Vec3d(0,0,.06)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye,target,Gf.Vec3d(0,0,1)).GetInverse())
        camera.CreateFocalLengthAttr(24.);camera.CreateClippingRangeAttr(Gf.Vec2f(.001,1000.))
        # Supplemental white light changes only rendering, never recorded states.
        fill=UsdLux.DomeLight.Define(stage,'/World/RawVideoFill');fill.CreateIntensityAttr(300.)
        rep.orchestrator.set_capture_on_play(False)
        rp=rep.create.render_product(str(camera.GetPath()),(a.width,a.height))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([rp])
        for _ in range(12):app.update()
        end=min(a.end_seconds,float(d['time'][-1]))
        times=np.arange(a.start_seconds,end+1e-9,1/a.fps)
        if not len(times):raise ValueError('No capture frames')
        sampled=[]
        for f,sec in enumerate(times):
            k=int(np.argmin(abs(d['time']-sec)));apply(k)
            rep.orchestrator.step(rt_subframes=2,pause_timeline=True)
            pixels=np.asarray(rgb.get_data())
            if pixels.shape[:2]!=(a.height,a.width):raise RuntimeError(f'RGB unavailable: {pixels.shape}')
            Image.fromarray(pixels[:,:,:3]).save(frames_dir/f'{f:06d}.png')
            sampled.append(float(d['time'][k]))
            if f%48==0:print(f'Native RGB {f}/{len(times)} t={d["time"][k]:.3f}s',flush=True)
        subprocess.run(['/usr/bin/ffmpeg','-v','error','-framerate',str(a.fps),'-i',str(frames_dir/'%06d.png'),
            '-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(a.output/'raw.mp4')],check=True)
        save('provenance.json',dict(renderer='Isaac Sim native RTX / Replicator RGB',physics_rerun=False,
            uncomposited=True,overlays=False,source_run=str(a.run.resolve()),source_scene=str(source.resolve()),
            source_scene_sha256=sha(source),trajectory_sha256=sha(a.run/'trajectory.npz'),case=a.case,
            passed=result['info']['passed'][i],fps=a.fps,width=a.width,height=a.height,frame_times_seconds=sampled,
            coordinate_frame='original object axes before six-direction environment rotation',
            camera_eye=list(eye),camera_target=list(target),note='Saved-state native replay, not live dynamics recording; SDF input triangles; no resimulation'))
        save('status.json',dict(status='complete',frames=len(sampled),video=str((a.output/'raw.mp4').resolve())))
        print(a.output/'raw.mp4',flush=True)
    except BaseException as exc:
        import traceback
        save('status.json',dict(status='failed',message=str(exc),traceback=traceback.format_exc()));raise
    finally:app.close()

if __name__=='__main__':main()
