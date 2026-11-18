"""Versioned candidate removal and macro surface-patch contact frames."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import trimesh
from admit_l20_middle_contact import load_hand
from robust_surface_query import nearest_surface,closest_triangle_points
from contact_frame_display import contact_basis
ROOT=Path(__file__).resolve().parents[1]
REMOVED={54,55,73,76}
RULES={33:'Xc',50:'-Xc',65:'-Xc',74:'-Xc',79:'-Xc',67:'Yc'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def patch_normal(mesh,q,radius,anchor=None):
    ids=np.array(list(mesh.triangles_tree.intersection(np.r_[q-radius,q+radius])),dtype=int)
    if not len(ids):raise ValueError('No surface neighbourhood')
    tri=mesh.triangles[ids];closest=closest_triangle_points(tri,np.broadcast_to(q,(len(ids),3)))
    dist=np.linalg.norm(closest-q,axis=1);keep=dist<=radius
    ids=ids[keep];dist=dist[keep];normals=mesh.face_normals[ids]
    weight=np.minimum(mesh.area_faces[ids],np.pi*radius**2)*np.exp(-.5*(dist/(radius*.5))**2)
    usable=weight>1e-20;weight=weight[usable];normals=normals[usable];ids=ids[usable]
    if not len(ids):raise ValueError('Degenerate neighbourhood')
    seeds=normals[np.argsort(weight)[-64:]]
    clusters=(seeds@normals.T)>=np.cos(np.deg2rad(35))
    support=clusters@weight;conflict=False
    if anchor is not None:
        candidates=(seeds@np.asarray(anchor))>=np.cos(np.deg2rad(60))
        if candidates.any():support=np.where(candidates,support,-1)
        else:conflict=True
    cluster=clusters[int(np.argmax(support))]
    n=(weight[cluster,None]*normals[cluster]).sum(0);n/=np.linalg.norm(n)
    ratio=float(weight[cluster].sum()/weight.sum())
    return n,dict(radius_mm=radius*1000,neighbour_faces=len(ids),dominant_area_fraction=ratio,needs_review=ratio<.6 or conflict,anchor_conflict=conflict,
        method='Area-weighted dominant 35deg normal cluster within radius; capped face area; not certified contact geometry')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-frames',type=Path,default=ROOT/'data/reports/l20_every_contact_frame_no_webgl_20261117/all_contact_frames.json')
    p.add_argument('--patch-radius-mm',type=float,default=2.)
    p.add_argument('--anchor-radius-mm',type=float,default=8.)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new version directory')
    if not np.isfinite([a.patch_radius_mm,a.anchor_radius_mm]).all() or min(a.patch_radius_mm,a.anchor_radius_mm)<=0:p.error('Invalid patch/anchor radius')
    reference={f['global_id']:f for f in json.loads(a.reference_frames.read_text())}
    hand,urdf=load_hand();original=json.loads((urdf.parent/'contact_points.json').read_text())
    if len(reference)!=hand.n_contact_candidates:p.error('Wrong frozen reference point count')
    anchors={}
    for stable in list(RULES)+[211,219,228,231]:
        ref=reference[stable];rule=RULES.get(stable,'Zc');axis=rule.lstrip('-');direction=np.asarray(ref[axis])*(-1 if rule.startswith('-') else 1)
        anchors.setdefault(ref['parent_link'],[]).append((stable,np.asarray(ref['origin_P_m']),direction))
    a.output.mkdir(parents=True);frames=[];mapping=[];points_out={k:[] for k in original};old_id=0;runtime=0
    for link,data in hand.mesh.items():
        vertices=data['vertices'].numpy();faces=data['faces'].numpy();mesh=trimesh.Trimesh(vertices,faces,process=False)
        pts=data['contact_candidates'].numpy()
        if not len(pts):continue
        q,dist,nearest=nearest_surface(mesh,pts)
        for local,(point,cp,gap,face) in enumerate(zip(pts,q,dist,nearest)):
            stable=old_id;old_id+=1
            if reference[stable]['parent_link']!=link or not np.allclose(point,reference[stable]['origin_P_m'],atol=1e-9):raise ValueError('Frozen IDs no longer match source')
            if stable in REMOVED:
                mapping.append(dict(stable_id=stable,source_link=link,source_local_id=local,removed=True,runtime_id=None));continue
            nearby=sorted(anchors.get(link,[]),key=lambda x:np.linalg.norm(x[1]-point))
            anchor=nearby[0] if nearby and np.linalg.norm(nearby[0][1]-point)<=a.anchor_radius_mm/1000 else None
            n,quality=patch_normal(mesh,cp,a.patch_radius_mm/1000,anchor[2] if anchor else None)
            quality.update(anchor_stable_id=anchor[0] if anchor else None,anchor_distance_mm=float(np.linalg.norm(anchor[1]-point)*1000) if anchor else None)
            raw=mesh.face_normals[face];rule=RULES.get(stable);normal_source='surface_patch_estimate_unreviewed'
            if rule:
                axis=rule.lstrip('-');n=np.asarray(reference[stable][axis])*(-1 if rule.startswith('-') else 1);normal_source='user_confirmed_20261117'
            elif stable in [211,219,228,231]:
                n=np.asarray(reference[stable]['Zc']);normal_source='preserved_user_confirmed_20261116';rule='reference_Zc'
            # Stable tangent convention: link +Z projected onto the new tangent plane.
            n,t1,t2=contact_basis(n,[0,0,1])
            newlocal=len(points_out[link]);points_out[link].append(original[link][local])
            frame=dict(global_id=stable,stable_id=stable,runtime_id=runtime,parent_link=link,local_id=newlocal,source_local_id=local,
                origin_P_m=point.tolist(),surface_projection_Q_m=cp.tolist(),surface_offset_mm=float(gap*1000),Xc=t1.tolist(),Yc=t2.tolist(),Zc=n.tolist(),geometric_N=raw.tolist(),normal_source=normal_source,user_rule=rule,
                patch_quality=quality,needs_review=normal_source=='surface_patch_estimate_unreviewed',
                convention='Xc=T1 red,Yc=T2 green,Zc=N blue; stable old P labels; runtime_id is new generation index')
            frames.append(frame);mapping.append(dict(stable_id=stable,runtime_id=runtime,source_link=link,source_local_id=local,new_local_id=newlocal,removed=False));runtime+=1
        print(link,'done',flush=True)
    if old_id!=420 or runtime!=416:raise ValueError('Unexpected source/removal count')
    (a.output/'contact_points.json').write_text(json.dumps(points_out,indent=2)+'\n')
    (a.output/'contact_frames.json').write_text(json.dumps(frames,indent=2)+'\n');(a.output/'id_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    import xml.etree.ElementTree as ET
    files=[urdf,urdf.parent/'contact_points.json',ROOT/'grasp_generation/utils/hand_model.py']+[(urdf.parent/m.get('filename')).resolve() for m in ET.parse(urdf).getroot().findall('link/visual/geometry/mesh')]
    manifest=dict(version='l20-surface-patch-anchors-v3-20261117',source_reference=str(a.reference_frames.resolve()),source_reference_sha256=sha(a.reference_frames),removed_stable_ids=sorted(REMOVED),manual_rules={str(k):v for k,v in RULES.items()},point_count=runtime,anchor_radius_mm=a.anchor_radius_mm,
        asset_sha256={str(f):sha(f) for f in files},generated_sha256={f:sha(a.output/f) for f in ['contact_points.json','contact_frames.json','id_mapping.json']},patch_radius_mm=a.patch_radius_mm,
        implementation_sha256=sha(Path(__file__)),note='New library, not activated in old generator. Stable labels differ from new dense runtime IDs. Automatic normals require review; not dynamics or region certification.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Complete:',a.output.resolve(),'416 points')


if __name__=='__main__':main()
