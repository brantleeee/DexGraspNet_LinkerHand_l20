"""RGB capture during an actual dynamics run; rendering must not advance physics."""
import json,subprocess
from pathlib import Path
import numpy as np


class LiveDynamicsVideo:
    def __init__(self,env,out,case='gy+',width=960,height=720):
        import omni.replicator.core as rep
        from pxr import UsdGeom,Gf
        self.env=env;self.out=Path(out);self.frames=self.out/'video_frames';self.frames.mkdir()
        self.times=[];self.rep=rep;self.width=width;self.height=height
        from isaacsim_grasp.environment import CASES
        self.index=list(env.case_ids).index(CASES.index(case));self.case=case
        r=env.rot[self.index];o=env.origins[self.index]
        eye=o+r@np.array([.4,-.32,.3]);target=o+r@np.array([0,0,.06]);up=r@np.array([0,0,1.])
        cam=UsdGeom.Camera.Define(env.stage,'/World/LiveValidationCamera')
        cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(*up)).GetInverse())
        cam.CreateFocalLengthAttr(24.);cam.CreateClippingRangeAttr(Gf.Vec2f(.001,1000.))
        rep.orchestrator.set_capture_on_play(False)
        self.product=rep.create.render_product(str(cam.GetPath()),(width,height))
        self.rgb=rep.AnnotatorRegistry.get_annotator('rgb');self.rgb.attach([self.product])
        # SimulationContext.render explicitly disables playSimulations during app update.
        for _ in range(5):self.render_checked()

    def render_checked(self):
        e=self.env;before=[np.asarray(v).copy() for v in e.state()];clock=e.sim.current_time
        e.sim.render()
        import carb.settings
        settings=carb.settings.get_settings()
        old=settings.get('/app/player/playSimulations')
        try:
            settings.set('/app/player/playSimulations',False)
            self.rep.orchestrator.step(rt_subframes=1,pause_timeline=False,delta_time=0.0)
        finally:settings.set('/app/player/playSimulations',old)
        after=e.state()
        if e.sim.current_time!=clock or any(not np.array_equal(a,b) for a,b in zip(before,after)):
            raise RuntimeError('Rendering changed physical state/time; refusing to mislabel capture')

    def capture(self):
        from PIL import Image
        self.render_checked()
        rgb=np.asarray(self.rgb.get_data())
        if rgb.shape[:2]!=(self.height,self.width):raise RuntimeError(f'RGB not ready: {rgb.shape}')
        Image.fromarray(rgb[:,:,:3]).save(self.frames/f'{len(self.times):06d}.png')
        self.times.append(self.env.physics_step*self.env.dt)

    def finish(self):
        if len(self.times)!=391 or not np.allclose(np.diff(self.times),1/30,atol=1e-8):
            raise RuntimeError('Expected complete 13-second 30fps live capture')
        subprocess.run(['/usr/bin/ffmpeg','-v','error','-framerate','30','-i',str(self.frames/'%06d.png'),
            '-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(self.out/'validation_live.mp4')],check=True)
        audit=dict(source='Isaac Sim live dynamics validation + RTX RGB',physics_enabled=True,saved_state_replay=False,
            case=self.case,frames=len(self.times),fps=30,width=self.width,height=self.height,
            sample_times_seconds=self.times,render_extra_physics_steps=0,
            note='Camera recorded the running physics stage; no state playback or compositing. Each render checked time and poses unchanged.')
        (self.out/'video_audit.json').write_text(json.dumps(audit,indent=2))
        # SimulationApp.close releases render resources after results are saved.
        return audit
