"""Replay saved L20 states as USD animation inside Isaac Sim, without physics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from validate_l20_isaacsim import rotation, axis_rotation, transform
from l20_object_catalog import OBJECTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('data/experiments/l20_isaacsim_acquire_demo_20261104'))
    p.add_argument('--object',choices=OBJECTS,default='mug')
    p.add_argument('--candidate',type=int,default=20,help='Original NPY index')
    p.add_argument('--case',choices=['gx+','gx-','gy+','gy-','gz+','gz-'],default='gz-')
    p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--paused',action='store_true')
    p.add_argument('--no-loop',action='store_true')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--test-seconds',type=float,default=0.,help='Automatically close after this many seconds; 0 holds window')
    p.add_argument('--export-usd',type=Path,help='Save a standalone animation USD with physics disabled')
    p.add_argument('--status-file',type=Path)
    a=p.parse_args()
    if a.speed<=0:p.error('--speed must be positive')
    if a.headless and a.test_seconds<=0:p.error('--headless requires a positive --test-seconds')
    if (a.results/'summary_all.json').exists():
        rows=json.loads((a.results/'summary_all.json').read_text())['rows']
        row=next((r for r in rows if r['object']==a.object and r['candidate']==a.candidate),None)
        if row is None or not row['completed']:p.error('Completed candidate not found')
        result=Path(row['output'])
    else:
        result=a.results
    s=json.loads((result/'summary.json').read_text());m=s['manifest']
    c=next(c for c in m['candidates'] if c['index']==a.candidate)
    if m['object']!=a.object:p.error('Result object does not match --object')
    dt=s['parameters']['dt'];cases=[r['case'] for r in s['results']]
    if a.case not in cases:p.error('Case not available')
    state={'status':'starting','result':str(result.resolve()),'case':a.case,'speed':a.speed,'physics_enabled':False}
    def save(**values):
        state.update(values)
        if a.status_file:
            a.status_file.parent.mkdir(parents=True,exist_ok=True)
            a.status_file.write_text(json.dumps(state,indent=2))
    save()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':a.headless,'multi_gpu':False,'sync_loads':False,'create_new_stage':False,
                       'limit_cpu_threads':8,'anti_aliasing':2,'denoiser':False})
    try:
        import omni.usd
        import omni.timeline
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,Sdf,Usd
        from isaacsim.core.utils.viewports import set_camera_view
        context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
        # Copy into an anonymous stage: playback never edits the validation scene.
        stage.GetRootLayer().TransferContent(Sdf.Layer.FindOrOpen(str((result/'scene.usda').resolve())))
        timeline=omni.timeline.get_timeline_interface();timeline.stop();timeline.set_auto_update(False)
        colliders=[]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='sdf':raise ValueError('Non-SDF source collider')
                colliders.append(str(prim.GetPath()))
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        if len(colliders)!=23:raise ValueError('Expected 23 SDF source colliders')
        for path in ['/physicsScene','/World/Hand/joints','/World/Hand/world_fixed']:
            prim=stage.GetPrimAtPath(path)
            if prim:prim.SetActive(False)
        save(status='building_animation',source_colliders=len(colliders),source_approximation='sdf')
        linkops={}
        for link in m['links']:
            ops={op.GetOpName():op for op in UsdGeom.Xformable(stage.GetPrimAtPath('/World/Hand/'+link['name'])).GetOrderedXformOps()}
            linkops[link['name']]=(ops['xformOp:translate'],ops['xformOp:orient'])
        objectops={op.GetOpName():op for op in UsdGeom.Xformable(stage.GetPrimAtPath('/World/Object')).GetOrderedXformOps()}
        wrist=c['qpos'];base=transform([wrist['WRJT'+v] for v in 'xyz'],rotation([wrist['WRJR'+v] for v in 'xyz']))
        ordered=[];known={'base_link'};pending=list(m['joints'])
        while pending:
            before=len(pending)
            for joint in pending[:]:
                if joint['parent'] in known:
                    ordered.append(joint);known.add(joint['child']);pending.remove(joint)
            if len(pending)==before:raise ValueError('Invalid URDF tree')
        origins={j['name']:transform(j['xyz'],rotation(j['rpy'])) for j in ordered}
        frames=0;duration=0.;speed=a.speed;loading=False;last_case=a.case;phase_codes=np.array([2])
        def set_speed(value):
            nonlocal speed
            speed=max(.05,float(value))
        def load_case(case):
            nonlocal frames,duration,last_case,loading,phase_codes
            loading=True;timeline.pause();save(status='building_animation',case=case)
            data=np.load(result/f'trajectory_{case}.npz')
            names=data['dof_names'].tolist();nameindex={n:i for i,n in enumerate(names)}
            initialq=data['initial_joint_positions'] if 'initial_joint_positions' in data else [c['targets'][n] for n in names]
            q=np.vstack([initialq,data['joint_positions']])
            phase_codes=np.r_[0,data['phase']] if 'phase' in data else np.full(len(q),2)
            pos=np.vstack([np.zeros(3),data['object_position']]);ori=np.vstack([[1,0,0,0],data['object_quaternion']])
            frames=len(q);duration=float(data['time'][-1]);last_case=case
            with Sdf.ChangeBlock():
                for pair in linkops.values():
                    for op in pair:
                        for code in op.GetTimeSamples():op.GetAttr().ClearAtTime(code)
                for op in objectops.values():
                    for code in op.GetTimeSamples():op.GetAttr().ClearAtTime(code)
                for i,angles in enumerate(q):
                    fk={'base_link':base}
                    for joint in ordered:
                        angle=0. if joint['type']=='fixed' else angles[nameindex[joint['name']]]
                        fk[joint['child']]=fk[joint['parent']]@origins[joint['name']]@transform([0,0,0],axis_rotation(joint['axis'],angle))
                    for name,t in fk.items():
                        trans,orient=linkops[name]
                        trans.Set(Gf.Vec3d(*t[:3,3]),i)
                        orient.Set(Gf.Quatf(Gf.Matrix3d(*t[:3,:3].T.flatten().tolist()).ExtractRotation().GetQuat()),i)
                    objectops['xformOp:translate'].Set(Gf.Vec3d(*pos[i]),i)
                    objectops['xformOp:orient'].Set(Gf.Quatf(float(ori[i,0]),Gf.Vec3f(*ori[i,1:])),i)
            stage.SetStartTimeCode(0);stage.SetEndTimeCode(frames-1)
            stage.SetTimeCodesPerSecond(1/dt);stage.SetFramesPerSecond(1/dt)
            timeline.set_time_codes_per_second(1/dt)
            timeline.set_start_time(0);timeline.set_end_time(duration);timeline.set_current_time(0)
            timeline.set_looping(not a.no_loop)
            loading=False;save(status='ready',case=case,frames=frames,duration_seconds=duration,
                               protocol=s['parameters'].get('protocol','direct'))
        load_case(a.case)
        def reset_camera():
            cache=UsdGeom.BBoxCache(Usd.TimeCode(0),[UsdGeom.Tokens.default_])
            box=cache.ComputeWorldBound(stage.GetPrimAtPath('/World/Hand')).ComputeAlignedRange()
            box.UnionWith(cache.ComputeWorldBound(stage.GetPrimAtPath('/World/Object')).ComputeAlignedRange())
            centre=np.array(box.GetMidpoint());size=float(np.linalg.norm(box.GetSize()))
            away=-base[:3,3].copy();away[2]=0
            if np.linalg.norm(away)<1e-6:away=np.array([1.,1.,0.])
            away/=np.linalg.norm(away);away[2]=.55;away/=np.linalg.norm(away)
            set_camera_view(eye=centre+away*size*1.6,target=centre)
        reset_camera()
        if a.export_usd:
            if a.export_usd.exists():raise ValueError('Export path must be new')
            a.export_usd.parent.mkdir(parents=True,exist_ok=True)
            stage.GetRootLayer().Export(str(a.export_usd.resolve()))
            save(exported_usd=str(a.export_usd.resolve()))
        ui_refs=[];updating=False
        if not a.headless:
            import omni.ui as ui
            window=ui.Window('L20 Recorded Grasp Replay',width=390,height=285)
            with window.frame:
                with ui.VStack(spacing=8):
                    ui.Label(f'{a.object} | candidate {a.candidate} | scale {m["scale"]}')
                    ui.Label('Saved states only. Physics disabled. SDF input meshes.')
                    combo=ui.ComboBox(cases.index(a.case),*cases)
                    combo.model.add_item_changed_fn(lambda model,item:load_case(cases[model.get_item_value_model().as_int]))
                    with ui.HStack(height=25):
                        ui.Button('Play / Pause',clicked_fn=lambda:timeline.pause() if timeline.is_playing() else timeline.play())
                        ui.Button('Restart',clicked_fn=lambda:timeline.set_current_time(0))
                        ui.Button('Reset camera',clicked_fn=reset_camera)
                    with ui.HStack(height=22):
                        ui.Label('Speed',width=55);speedmodel=ui.SimpleFloatModel(a.speed)
                        ui.FloatSlider(model=speedmodel,min=.1,max=2.)
                        speedmodel.add_value_changed_fn(lambda model:set_speed(model.as_float))
                    with ui.HStack(height=22):
                        ui.Label('Time (s)',width=65);seekmodel=ui.SimpleFloatModel(0.)
                        ui.FloatSlider(model=seekmodel,min=0.,max=duration)
                        def seek_changed(model):
                            if not updating and not loading:timeline.set_current_time(model.as_float)
                        seekmodel.add_value_changed_fn(seek_changed)
                    with ui.HStack(height=22):
                        for label,path in [('Hand','/World/Hand'),('Object','/World/Object')]:
                            model=ui.SimpleBoolModel(True);ui.CheckBox(model=model,width=24);ui.Label(label)
                            def visibility(value,path=path):
                                imageable=UsdGeom.Imageable(stage.GetPrimAtPath(path))
                                imageable.MakeVisible() if value.as_bool else imageable.MakeInvisible()
                            model.add_value_changed_fn(visibility);ui_refs.append(model)
                    label=ui.Label('')
            ui_refs.extend([window,combo,speedmodel,seekmodel,label])
        timeline.set_looping(not a.no_loop)
        if not a.paused:timeline.play()
        started=time.monotonic();last_save=started;last_tick=started;max_time=0.;max_frame=0.
        print(f'Replaying {a.object} candidate {a.candidate}: {result}, {frames} frames; physics OFF',flush=True)
        while app.is_running():
            app.update()
            now=time.monotonic();elapsed=min(now-last_tick,.1);last_tick=now
            current=timeline.get_current_time()
            if timeline.is_playing():
                current+=elapsed*speed
                if current>duration:
                    if not a.no_loop:current%=duration
                    else:current=duration;timeline.pause()
                timeline.set_current_time(current)
            max_time=max(max_time,current);max_frame=max(max_frame,current/dt)
            if ui_refs:
                phase=['CLOSE (g=0)','SETTLE (g=0)','GRAVITY'][int(phase_codes[min(len(phase_codes)-1,round(current/dt))])]
                updating=True;seekmodel.set_value(min(current,duration));label.text=f'{phase} | {current:.3f} / {duration:.3f} s';updating=False
            if time.monotonic()-last_save>1:
                phase_name=['close_zero_g','settle_zero_g','gravity_test'][int(phase_codes[min(len(phase_codes)-1,round(current/dt))])]
                save(status='playing' if timeline.is_playing() else 'paused',case=last_case,current_saved_time=current,max_frame=max_frame,phase=phase_name)
                last_save=time.monotonic()
            if a.test_seconds>0 and time.monotonic()-started>=a.test_seconds:break
            time.sleep(.005)
        timeline.stop()
        save(status='closed',max_saved_time=max_time,max_frame=max_frame,physics_enabled=False)
    except BaseException as exc:
        import traceback
        save(status='error',error=str(exc),traceback=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
        raise
    finally:app.close()


if __name__=='__main__':main()
