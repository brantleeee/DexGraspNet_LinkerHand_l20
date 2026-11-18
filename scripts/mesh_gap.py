"""Truncated exact triangle-surface gap; original meshes, no convex proxy."""
import numpy as np
from check_l20_middle_self_collision import triangle_overlap


def triangle_distance(a,b):
    from trimesh.triangles import closest_point
    result=np.full(len(a),np.inf)
    for x,y in [(a,b),(b,a)]:
        for i in range(3):
            q=closest_point(y,x[:,i])
            result=np.minimum(result,np.linalg.norm(q-x[:,i],axis=1))
    for i in range(3):
        p=a[:,i];u=a[:,(i+1)%3]-p
        for j in range(3):
            q=b[:,j];v=b[:,(j+1)%3]-q;w=p-q
            aa=(u*u).sum(1);bb=(u*v).sum(1);cc=(v*v).sum(1)
            dd=(u*w).sum(1);ee=(v*w).sum(1);den=aa*cc-bb*bb
            s=(bb*ee-cc*dd)/np.maximum(den,1e-30);t=(aa*ee-bb*dd)/np.maximum(den,1e-30)
            valid=(den>1e-24)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
            d=np.linalg.norm(w+s[:,None]*u-t[:,None]*v,axis=1)
            result=np.minimum(result,np.where(valid,d,np.inf))
            for point,start,edge in [(p,q,v),(p+u,q,v),(q,p,u),(q+v,p,u)]:
                alpha=np.clip(((point-start)*edge).sum(1)/np.maximum((edge*edge).sum(1),1e-30),0,1)
                result=np.minimum(result,np.linalg.norm(point-start-alpha[:,None]*edge,axis=1))
    result[triangle_overlap(a,b)]=0.
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite triangle distance; refusing to certify mesh gap')
    return result


def truncated_gap(a,b,margin):
    """Returns min(actual surface gap, margin), up to float/touch tolerance."""
    if np.any(a.bounds[0]-b.bounds[1]>margin) or np.any(b.bounds[0]-a.bounds[1]>margin):
        return margin
    from rtree import index
    tree=index.Index(properties=index.Property(dimension=3))
    bt=b.triangles
    for i,t in enumerate(bt):tree.insert(i,tuple(np.r_[t.min(0),t.max(0)]))
    best=margin
    for t in a.triangles:
        ids=list(tree.intersection(tuple(np.r_[t.min(0)-best,t.max(0)+best])))
        for start in range(0,len(ids),2048):
            part=bt[ids[start:start+2048]]
            best=min(best,float(triangle_distance(np.broadcast_to(t,part.shape),part).min()))
            if best<=1e-9:return 0.
    return best


def contained(source,target):
    """Only certify a positive containment for closed, consistently wound target."""
    import trimesh
    if np.any(source.bounds[1]<target.bounds[0]) or np.any(target.bounds[1]<source.bounds[0]):return False
    t=trimesh.Trimesh(target.vertices,target.faces,process=True)
    if not(t.is_watertight and t.is_winding_consistent and t.volume>0):return None
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e=source.edges_unique
    graph=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(source.vertices),len(source.vertices)))
    _,labels=connected_components(graph,directed=False)
    reps=source.vertices[np.unique(labels,return_index=True)[1]]
    return bool(t.contains(reps).any())
