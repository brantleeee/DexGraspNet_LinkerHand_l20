"""Geometric facet-normal witnesses, including edge ambiguity."""
import numpy as np
import trimesh


def normal_witness(mesh,point,closest,face_index):
    tri=mesh.triangles[face_index]
    e1,e2=tri[1]-tri[0],tri[2]-tri[0]
    normal=np.cross(e1,e2);normal/=np.linalg.norm(normal)
    bary=trimesh.triangles.points_to_barycentric(tri[None],np.asarray(closest)[None])[0]
    zero=np.flatnonzero(np.abs(bary)<1e-5)
    welded=mesh.copy();welded.merge_vertices()
    neighbours=set()
    for opposite in zero:
        edge=np.delete(welded.faces[face_index],opposite)
        match=np.any(welded.faces==edge[0],axis=1)&np.any(welded.faces==edge[1],axis=1)
        neighbours.update(np.flatnonzero(match).tolist())
    neighbours.discard(int(face_index))
    return dict(face_index=int(face_index),triangle=tri.tolist(),normal=normal.tolist(),barycentric=bary.tolist(),
        projection_on_edge=bool(len(zero)),projection_on_vertex=bool(len(zero)>1),
        normal_edge_cosines=[float(np.dot(normal,e)/np.linalg.norm(e)) for e in [e1,e2]],
        candidate_surface_distance_mm=float(np.linalg.norm(np.asarray(point)-closest)*1000),
        adjacent_faces=[dict(face_index=int(i),triangle=mesh.triangles[i].tolist(),normal=mesh.face_normals[i].tolist(),
            angle_degrees=float(np.degrees(np.arccos(np.clip(normal@mesh.face_normals[i],-1,1))))) for i in sorted(neighbours)],
        interpretation='Selected facet normal, not a unique physical contact normal at an edge/vertex.')
