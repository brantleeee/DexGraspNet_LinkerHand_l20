"""Read-only per-link and ragged contact data in original object coordinates."""
from pathlib import Path
import numpy as np
from .diagnostics import PhysicsDiagnostics
from .isolated import finger_name,contact_events

FINGERS=['thumb','index','middle','ring','pinky','palm']


class FingerDiagnostics(PhysicsDiagnostics):
    def __init__(self,env):
        self.links=env.ref['hand_links'];self.point_frames=[];self.friction_frames=[]
        super().__init__(env)

    def begin(self):
        self.point_frames=[];self.friction_frames=[]
        super().begin()

    def capture(self,initial=False):
        super().capture(initial)
        e=self.env;counts=np.zeros((e.n,len(self.links)),int);forces=np.zeros((e.n,len(self.links),3))
        scalar=np.zeros((e.n,len(self.links)));points_out=[];friction_out=[]
        if not initial:
            normal,points,normals,separation,counts,starts=e.obj.get_contact_force_data(dt=e.dt)
            counts=np.asarray(counts).reshape(e.n,-1);starts=np.asarray(starts).reshape(e.n,-1)
            if counts.shape!=(e.n,len(self.links)):raise RuntimeError('Contact filter/link ordering or count differs')
            forces=np.asarray(e.obj.get_contact_force_matrix(dt=e.dt))
            forces=np.einsum('nji,nkj->nki',e.rot,forces)
            normal=np.asarray(normal).reshape(-1);points=np.asarray(points);normals=np.asarray(normals);separation=np.asarray(separation).reshape(-1)
            for i in range(e.n):
                for link in range(len(self.links)):
                    start=int(starts[i,link]);count=int(counts[i,link]);ids=np.arange(start,start+count)
                    if not count:continue
                    scalar[i,link]=np.sum(abs(normal[ids]))
                    p=(points[ids]-e.origins[i])@e.rot[i];n=normals[ids]@e.rot[i]
                    points_out.append(np.column_stack([np.full(count,e.physics_step),np.full(count,i),np.full(count,link),p,n,normal[ids],separation[ids]]))
            friction,anchors,fc,fs=e.obj.get_friction_data(dt=e.dt)
            fc=np.asarray(fc).reshape(e.n,-1);fs=np.asarray(fs).reshape(e.n,-1)
            if fc.shape!=counts.shape:raise RuntimeError('Friction filters differ from normal filters')
            for i in range(e.n):
                for link in range(len(self.links)):
                    start=int(fs[i,link]);count=int(fc[i,link]);ids=np.arange(start,start+count)
                    if count:
                        force=np.asarray(friction)[ids]@e.rot[i];p=(np.asarray(anchors)[ids]-e.origins[i])@e.rot[i]
                        friction_out.append(np.column_stack([np.full(count,e.physics_step),np.full(count,i),np.full(count,link),p,force]))
        finger_force=np.zeros((e.n,6));finger_count=np.zeros((e.n,6),int)
        for k,link in enumerate(self.links):
            f=FINGERS.index(finger_name(link));finger_force[:,f]+=scalar[:,k];finger_count[:,f]+=counts[:,k]
        self.frames[-1].update(link_contact_count=counts.copy(),link_normal_force_vector=forces.copy(),
            link_normal_force_magnitude_sum=scalar.copy(),finger_normal_force=finger_force,finger_contact_count=finger_count)
        self.point_frames.append(np.concatenate(points_out) if points_out else np.empty((0,11)))
        self.friction_frames.append(np.concatenate(friction_out) if friction_out else np.empty((0,9)))

    def finish(self,path):
        result=super().finish(path)
        contact_path=Path(path).with_name('contact_points.npz')
        points=np.concatenate(self.point_frames);friction=np.concatenate(self.friction_frames)
        if not np.isfinite(points).all() or not np.isfinite(friction).all():raise RuntimeError('Nonfinite contact details')
        np.savez_compressed(contact_path,normal_contacts=points,friction_anchors=friction,
            normal_columns=['physics_step','env','link','x','y','z','nx','ny','nz','normal_force_n','separation_m'],
            friction_columns=['physics_step','env','link','x','y','z','fx','fy','fz'],link_names=self.links,finger_names=FINGERS,
            coordinate_frame='original_object_frame_before_scene_rotation',
            normal_note='SDK sensor normal sign retained; vectors not relabeled as object-to-hand; friction anchors are not one-to-one with normal points')
        with np.load(path) as data:
            summaries=[]
            for i in range(self.env.n):
                summary={}
                for f,name in enumerate(FINGERS):
                    active=(data['finger_normal_force'][:,i,f]>.001)&(data['finger_contact_count'][:,i,f]>0)
                    summary[name]=contact_events(data['time'],active)
                summaries.append(summary)
        result.update(contact_points=str(contact_path),finger_contact_events=summaries)
        self.point_frames=[];self.friction_frames=[]
        return result
