"""Rebuild requested frames on explicit supporting triangles, preserving v4."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np,trimesh
from admit_l20_middle_contact import load_hand
from robust_surface_query import closest_triangle_points
from contact_frame_display import contact_basis
ROOT=Path(__file__).resolve().parents[1]
REMOVED={271,273,280,284}
LINKS={'middle_distal','ring_distal'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def supporting_face(mesh,point,tie_tolerance=1e-8):
    """Use a real facet, not the nearest-edge normal cone or disconnected average.

    When equally close faces meet, prefer the largest triangle deterministically,
    and expose every materially different tied normal for review.
    """
    point=np.asarray(point,dtype=float)
    ids=np.asarray(trimesh.proximity.nearby_faces(mesh,[point])[0],dtype=int)
    q=closest_triangle_points(mesh.triangles[ids],np.broadcast_to(point,(len(ids),3)))
    d=np.linalg.norm(q-point,axis=1);ties=np.flatnonzero(d<=d.min()+tie_tolerance)
    j=int(ties[np.argmax(mesh.area_faces[ids[ties]])]);face=int(ids[j])
    n=mesh.face_normals[face].copy();tri=mesh.triangles[face]
    angle=np.degrees(np.arccos(np.clip(mesh.face_normals[ids[ties]]@n,-1,1)))
    # Scale-normalize barycentric calculation for microfacets.
    scale=np.linalg.norm(tri-np.roll(tri,1,axis=0),axis=1).max()
    bary=trimesh.triangles.points_to_barycentric(((tri-tri[0])/scale)[None],((q[j]-tri[0])/scale)[None])[0]
    return q[j],n,dict(support_face_id=face,support_triangle_m=tri.tolist(),
        support_face_normal=n.tolist(),surface_offset_mm=float(d[j]*1000),
        barycentric=bary.tolist(),on_feature_edge=bool(bary.min()<1e-4),
        tied_face_count=len(ties),tied_normal_spread_deg=float(angle.max()),
        ambiguous_surface=bool(angle.max()>20),tie_tolerance_mm=tie_tolerance*1000,
        support_rule='nearest face; largest area only among distance ties; no across-edge averaging')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,default=ROOT/'data/hand_contacts/l20_contact_v4_20261117')
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    m=json.loads((a.previous/'manifest.json').read_text())
    for path,h in m['asset_sha256'].items():
        if sha(path)!=h:raise ValueError('Original source changed: '+path)
    for path,h in m['generated_sha256'].items():
        if sha(a.previous/path)!=h:raise ValueError('Previous library changed: '+path)
    old=json.loads((a.previous/'contact_frames.json').read_text());hand,_=load_hand()
    meshes={link:trimesh.Trimesh(data['vertices'].numpy(),data['faces'].numpy(),process=True)
        for link,data in hand.mesh.items() if link in LINKS or link=='thumb_metacarpals'}
    frames=[];points={k:[] for k in json.loads((a.previous/'contact_points.json').read_text())};mapping=[]
    removed=set(m['removed_stable_ids'])|REMOVED
    for prior in old:
        i=prior['stable_id']
        if i in removed:continue
        f=copy.deepcopy(prior);link=f['parent_link']
        if link in LINKS or i==337:
            mesh=meshes[link];q,n,evidence=supporting_face(mesh,f['origin_P_m'])
            n,t1,t2=contact_basis(n,[0,0,1])
            centroid=np.mean(evidence['support_triangle_m'],axis=0)
            closed=bool(mesh.is_volume and mesh.is_winding_consistent)
            probe=None
            if closed:
                center=mesh.bounds.mean(0);scaled=trimesh.Trimesh((mesh.vertices-center)*1000,mesh.faces,process=False)
                checks=scaled.contains((np.array([centroid+.00001*n,centroid-.00001*n])-center)*1000)
                probe=bool(not checks[0] and checks[1])
            evidence.update(closed_mesh=closed,outward_probe_pass=probe,probe_origin='support triangle centroid',probe_distance_mm=.01,
                old_normal_change_deg=float(np.degrees(np.arccos(np.clip(np.dot(prior['Zc'],n),-1,1)))),
                method='explicit supporting facet; no normal cone at nearest edge',dominant_area_fraction=None)
            f.update(Xc=t1.tolist(),Yc=t2.tolist(),Zc=n.tolist(),geometric_N=n.tolist(),
                surface_projection_Q_m=q.tolist(),surface_offset_mm=evidence['surface_offset_mm'],
                normal_source='explicit_supporting_facet_v5_unreviewed',previous_normal_source=prior['normal_source'],
                previous_Zc=prior['Zc'],patch_quality=evidence,needs_review=True,
                review_status='P337_intended_surface_pending' if i==337 else ('feature_edge_surface_choice_pending' if evidence['ambiguous_surface'] else 'facet_attached_region_pending'))
        f['runtime_id']=len(frames);f['local_id']=len(points[link]);points[link].append(f['origin_P_m']);frames.append(f)
    by_id={f['stable_id']:f for f in frames}
    for entry in json.loads((a.previous/'id_mapping.json').read_text()):
        entry=copy.deepcopy(entry);i=entry['stable_id'];entry['removed']=i in removed
        entry['runtime_id']=by_id[i]['runtime_id'] if i in by_id else None
        if i in by_id:entry['new_local_id']=by_id[i]['local_id']
        else:entry.pop('new_local_id',None)
        mapping.append(entry)
    a.output.mkdir(parents=True)
    for filename,data in [('contact_points.json',points),('contact_frames.json',frames),('id_mapping.json',mapping)]:
        (a.output/filename).write_text(json.dumps(data,indent=2)+'\n')
    manifest=dict(version='l20-facet-attached-v5-20261117',previous=str(a.previous.resolve()),previous_manifest_sha256=sha(a.previous/'manifest.json'),
        point_count=len(frames),removed_stable_ids=sorted(removed),new_removals=sorted(REMOVED),asset_sha256=m['asset_sha256'],
        rebuilt_links=sorted(LINKS),rebuilt_stable_ids=[337],positions_changed=False,
        previous_manual_normals_reestimated_on_requested_links=True,implementation_sha256=sha(Path(__file__)),
        generated_sha256={f:sha(a.output/f) for f in ['contact_points.json','contact_frames.json','id_mapping.json']},
        note='Facet attachment does not certify intended grasp face. P337 needs user direction/region clarification. Old library and default generator preserved.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Complete:',a.output.resolve(),len(frames),'points')


if __name__=='__main__':main()
