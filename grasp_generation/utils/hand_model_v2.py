"""Versioned raw-mesh distance model; preserve the legacy module and its asset hash."""
import trimesh
import torch
from torchsdf import index_vertices_by_faces,compute_sdf
from .hand_model import HandModel as LegacyHandModel


class HandModel(LegacyHandModel):
    def __init__(self,*args,force_mesh_sdf=True,penetration_geometry=None,**kwargs):
        geometry=penetration_geometry or ('raw-mesh' if force_mesh_sdf else 'legacy-convex')
        if geometry not in ['raw-mesh','legacy-convex']:raise ValueError('Unknown penetration geometry')
        super().__init__(*args,force_mesh_sdf=force_mesh_sdf,**kwargs)
        self.penetration_geometry=geometry;self.geometry_audit=[]
        for link,m in self.mesh.items():
            previous=len(m['face_verts'])
            if geometry=='raw-mesh':m['face_verts']=index_vertices_by_faces(m['vertices'],m['faces'].long())
            raw=trimesh.Trimesh(m['vertices'].detach().cpu().numpy(),m['faces'].detach().cpu().numpy(),process=True)
            self.geometry_audit.append(dict(link=link,raw_faces=len(m['faces']),query_faces=len(m['face_verts']),legacy_hull_faces=previous,
                source_watertight=bool(raw.is_watertight),source_winding_consistent=bool(raw.is_winding_consistent),
                query_representation=geometry,closed_volume_surface=bool(raw.is_volume),sign_certified=False))

    def cal_distance(self,x):
        if self.penetration_geometry=='legacy-convex':return super().cal_distance(x)
        local_hand=(x-self.global_translation[:,None])@self.global_rotation;distances=[]
        for link,m in self.mesh.items():
            t=self.current_status[link].get_matrix()
            p=(local_hand-t[:,:3,3][:,None])@t[:,:3,:3]
            raw,sign,_,_=compute_sdf(p.reshape(-1,3),m['face_verts'])
            signed=raw.add(1e-8).sqrt()*(-sign)
            # A pseudo-normal on an open/creased mesh cannot certify inside.
            # Outside the original bounds, inside is geometrically impossible.
            lower=m['vertices'].amin(0);upper=m['vertices'].amax(0)
            outside=((p<lower-1e-7)|(p>upper+1e-7)).any(-1)
            signed=signed.reshape(x.shape[:2])
            distances.append(torch.where(outside,-signed.abs(),signed))
        return torch.stack(distances).max(0).values
