"""Scale-normalized closest-point queries for tiny metre-scale mesh facets."""
import numpy as np
import trimesh


def closest_triangle_points(triangles,points):
    triangles=np.asarray(triangles,dtype=np.float64);points=np.asarray(points,dtype=np.float64)
    origin=triangles[:,0].copy()
    scale=np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2).max(axis=1)
    if np.any(scale<=1e-15):raise ValueError('Degenerate zero-size triangle')
    q=trimesh.triangles.closest_point((triangles-origin[:,None])/scale[:,None,None],(points-origin)/scale[:,None])
    return q*scale[:,None]+origin


def nearest_surface(mesh,points):
    points=np.asarray(points,dtype=float)
    candidates=trimesh.proximity.nearby_faces(mesh,points)
    result=[];distances=[];faces=[]
    for point,ids in zip(points,candidates):
        ids=np.asarray(ids,dtype=int)
        q=closest_triangle_points(mesh.triangles[ids],np.broadcast_to(point,(len(ids),3)))
        d=np.linalg.norm(q-point,axis=1);i=int(np.argmin(d))
        result.append(q[i]);distances.append(d[i]);faces.append(ids[i])
    return np.asarray(result),np.asarray(distances),np.asarray(faces,dtype=int)
