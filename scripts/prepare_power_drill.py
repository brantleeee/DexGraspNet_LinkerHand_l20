"""Import bundled power-drill OBJ as a centered, unit-longest-edge closed mesh.
No hull/decomposition/decimation: weld seams, bridge matching bore rims, union solids.
"""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1]


def close_matching_bore(mesh):
    edges,counts=np.unique(np.sort(mesh.edges,axis=1),axis=0,return_counts=True)
    if np.any(counts>2):raise ValueError('Nonmanifold edges; cannot safely bridge bore')
    boundary=edges[counts==1];groups=trimesh.graph.connected_components(boundary)
    if len(groups)!=2 or len(groups[0])!=len(groups[1]):raise ValueError('Expected two matching bore rims')
    a,b=map(np.asarray,groups);va,vb=mesh.vertices[a],mesh.vertices[b]
    delta=vb.mean(0)-va.mean(0);axis=int(np.argmax(abs(delta)))
    if abs(delta[axis])<1e-6 or np.max(abs(np.delete(delta,axis)))>1e-6:raise ValueError('Bore rims not axial')
    if np.ptp(va[:,axis])>1e-6 or np.ptp(vb[:,axis])>1e-6:raise ValueError('Bore rims not planar')
    distances=np.linalg.norm((va+delta)[:,None]-vb[None,:],axis=2);indices=distances.argmin(1)
    if len(np.unique(indices))!=len(a) or distances[np.arange(len(a)),indices].max()>2e-6:raise ValueError('Rims do not correspond')
    mapping=dict(zip(a,b[indices]));faces=[]
    for i,j in boundary:
        if i in mapping and j in mapping:faces.extend([[int(i),int(j),int(mapping[j])],[int(i),int(mapping[j]),int(mapping[i])]])
    result=trimesh.Trimesh(mesh.vertices,np.vstack([mesh.faces,faces]),process=False)
    trimesh.repair.fix_normals(result,multibody=True)
    if not result.is_watertight:raise ValueError('Bore repair did not close mesh')
    return result,len(faces)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'assets/power-drill-obj.zip')
    p.add_argument('--output',type=Path,default=ROOT/'data/meshdata/power-drill')
    a=p.parse_args();digest=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    if a.output.exists():
        manifest=a.output/'import.json'
        if manifest.exists():
            old=json.loads(manifest.read_text());mesh=a.output/'coacd/decomposed.obj'
            if old['archive_sha256']==digest and old['prepared_mesh_sha256']==hashlib.sha256(mesh.read_bytes()).hexdigest():print('Reusing audited mesh:',mesh);return
        p.error('Output exists with different or incomplete data; use a new output directory')
    with zipfile.ZipFile(a.archive) as archive:
        # Read named members, never extract arbitrary archive paths.
        obj=archive.read('power-drill.obj');license_text=archive.read('LICENSE.txt')
        mesh=trimesh.load(io.BytesIO(obj),file_type='obj',force='mesh',process=False,skip_materials=True)
        originals={n:archive.read(n) for n in archive.namelist() if not n.endswith('/') and '/' not in n and '\\' not in n}
    if not np.isfinite(mesh.vertices).all():raise ValueError('Nonfinite vertices')
    raw_bounds=mesh.bounds.copy();raw_faces=len(mesh.faces)
    mesh.merge_vertices(merge_tex=True,merge_norm=True);mesh.update_faces(mesh.nondegenerate_faces());mesh.update_faces(mesh.unique_faces());mesh.remove_unreferenced_vertices()
    parts=mesh.split(only_watertight=False);repaired=[];added=0
    for part in parts:
        if not part.is_watertight:part,n=close_matching_bore(part);added+=n
        trimesh.repair.fix_normals(part,multibody=True)
        if not part.is_volume:raise ValueError('Invalid solid after seam repair')
        repaired.append(part)
    union=trimesh.boolean.union(repaired,engine='manifold',check_volume=True)
    if not union.is_volume or not np.isfinite(union.vertices).all():raise ValueError('Invalid union')
    center=union.bounds.mean(0);length=float(union.extents.max());union.vertices=(union.vertices-center)/length
    a.output.mkdir(parents=True);source=a.output/'source';source.mkdir()
    for name,content in originals.items():(source/name).write_bytes(content)
    (a.output/'coacd').mkdir();target=a.output/'coacd/decomposed.obj';union.export(target)
    record=dict(object_code='power-drill',archive=str(a.archive.resolve()),archive_sha256=digest,
        source_units='unspecified; physical size is explicitly selected with generation scale',raw_bounds=raw_bounds.tolist(),
        raw_extents=np.ptp(raw_bounds,axis=0).tolist(),raw_faces=raw_faces,source_components=len(parts),
        repaired_bore_triangles=added,operations=['weld coincident seams','remove degenerate/duplicate faces','bridge paired bore rims','manifold solid union','AABB center and unit longest edge'],
        convex_hull=False,convex_decomposition=False,decimation=False,normalized_extents=union.extents.tolist(),
        normalized_volume=float(union.volume),watertight=bool(union.is_watertight),winding_consistent=bool(union.is_winding_consistent),
        center_in_source_units=center.tolist(),longest_edge_in_source_units=length,
        prepared_vertices=len(union.vertices),prepared_faces=len(union.faces),prepared_mesh_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        note='coacd/decomposed.obj is the legacy loader filename only: this file is a closed nonconvex solid union, not CoACD output. Prepared coordinates are the shared generation/validation frame.')
    (a.output/'import.json').write_text(json.dumps(record,indent=2)+'\n');(a.output/'LICENSE.txt').write_bytes(license_text)
    print(json.dumps(record,indent=2));print('Prepared:',target)

if __name__=='__main__':main()
