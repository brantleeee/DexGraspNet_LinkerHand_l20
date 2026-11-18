"""Versioned user edits and outward-distance contact frames; no simulation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,trimesh
from admit_l20_middle_contact import load_hand
from robust_surface_query import nearest_surface
from contact_frame_display import contact_basis
ROOT=Path(__file__).resolve().parents[1]
NEW_REMOVALS={60,63,212,218,227,237}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def oriented_distance_normal(probe,closest,inside):
    direction=np.asarray(probe)-np.asarray(closest)
    length=np.linalg.norm(direction)
    if length<1e-12:raise ValueError('No probe separation; normal is undefined')
    return direction/length*(-1 if inside else 1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,default=ROOT/'data/hand_contacts/l20_surface_patch_v3_20261117')
    p.add_argument('--project-links',nargs='+',default=['ring_distal'])
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a new version directory')
    previous=json.loads((a.previous/'manifest.json').read_text())
    for path,h in previous['asset_sha256'].items():
        if sha(path)!=h:raise ValueError('Previous source asset changed: '+path)
    for path,h in previous['generated_sha256'].items():
        if sha(a.previous/path)!=h:raise ValueError('Previous contact library changed')
    old={x['stable_id']:x for x in json.loads((a.previous/'contact_frames.json').read_text())}
    removed=set(previous['removed_stable_ids'])|NEW_REMOVALS
    manual={i:np.asarray(f['Zc']) for i,f in old.items() if 'user_confirmed' in f['normal_source']}
    manual.update({14:-np.asarray(old[14]['Zc']),20:-np.asarray(old[20]['Zc']),297:np.asarray(old[297]['Yc'])})
    hand,urdf=load_hand();output_points={k:[] for k in json.loads((urdf.parent/'contact_points.json').read_text())};frames=[];mapping=[];offset=0;runtime=0
    if any(link not in hand.mesh for link in a.project_links):p.error('Unknown projection link')
    for link,data in hand.mesh.items():
        points=data['contact_candidates'].numpy();ids=list(range(offset,offset+len(points)));offset+=len(points)
        if not len(points):continue
        mesh=trimesh.Trimesh(data['vertices'].numpy(),data['faces'].numpy(),process=True)
        q,d,face=nearest_surface(mesh,points);closed=bool(mesh.is_volume and mesh.is_watertight and mesh.is_winding_consistent)
        center=mesh.bounds.mean(0);scaled=trimesh.Trimesh((mesh.vertices-center)*1000,mesh.faces,process=False)
        inside=scaled.contains((points-center)*1000) if closed else np.zeros(len(points),dtype=bool)
        proposed=[]
        for j,idx in enumerate(ids):
            if idx in removed:proposed.append(np.array([0.,0.,1.]));continue
            n=manual.get(idx)
            if n is None:
                n=oriented_distance_normal(points[j],q[j],bool(inside[j])) if d[j]>1e-12 else mesh.face_normals[face[j]]
            proposed.append(n/np.linalg.norm(n))
        proposed=np.asarray(proposed)
        # Query both sides of the actual surface at 0.05mm; inconclusive remains pending.
        if closed:
            plus=scaled.contains((q+.00005*proposed-center)*1000)
            minus=scaled.contains((q-.00005*proposed-center)*1000)
        normal_sources={}
        if closed:
            # At an edge the probe-to-Q vector can lie in the adjacent face.
            # Only replace an automatic estimate when a facet normal passes
            # the same two-sided test; never override a user's frozen rule.
            for j,idx in enumerate(ids):
                if idx in removed or idx in manual or (not plus[j] and minus[j]):continue
                alternative=mesh.face_normals[face[j]]
                checks=scaled.contains((np.array([q[j]+.00005*alternative,q[j]-.00005*alternative])-center)*1000)
                if not checks[0] and checks[1]:
                    proposed[j]=alternative;plus[j],minus[j]=checks
                    normal_sources[idx]='outward_facet_fallback_two_sided_checked'
        for j,idx in enumerate(ids):
            if idx in removed:mapping.append(dict(stable_id=idx,removed=True,runtime_id=None));continue
            n,t1,t2=contact_basis(proposed[j],[0,0,1]);position=q[j] if link in a.project_links else points[j]
            local=len(output_points[link]);output_points[link].append(position.tolist())
            probe_ok=bool(not plus[j] and minus[j]) if closed else None
            f=dict(global_id=idx,stable_id=idx,runtime_id=runtime,parent_link=link,local_id=local,source_local_id=j,
                original_P_m=points[j].tolist(),origin_P_m=position.tolist(),surface_projection_Q_m=q[j].tolist(),surface_offset_mm=float(np.linalg.norm(position-q[j])*1000),position_change_mm=float(np.linalg.norm(position-points[j])*1000),
                Xc=t1.tolist(),Yc=t2.tolist(),Zc=n.tolist(),geometric_N=mesh.face_normals[face[j]].tolist(),normal_source='user_confirmed_v4' if idx in manual else normal_sources.get(idx,'outward_distance_gradient_estimate'),
                needs_review=idx not in manual or probe_ok is False,
                review_status='outward_check_conflict' if probe_ok is False else ('open_mesh_orientation_unknown' if probe_ok is None else 'local_outward_check_passed'),
                original_probe_inside=bool(inside[j]) if closed else None,
                patch_quality=dict(method='outward distance gradient at original probe, inside sign flipped if closed',dominant_area_fraction=None,closed_mesh=closed,outward_probe_pass=probe_ok,probe_distance_mm=.05),
                convention='Stable P labels; Xc=T1 red/Yc=T2 green/Zc=N blue; original or explicitly projected point origin')
            frames.append(f);mapping.append(dict(stable_id=idx,runtime_id=runtime,new_local_id=local,source_link=link,source_local_id=j,removed=False));runtime+=1
        print(link,'closed',closed,'source probe inside',int(inside.sum()) if closed else 'unknown',flush=True)
    a.output.mkdir(parents=True)
    for name,data in [('contact_points.json',output_points),('contact_frames.json',frames),('id_mapping.json',mapping)]:
        (a.output/name).write_text(json.dumps(data,indent=2)+'\n')
    manifest=dict(version='l20-outward-distance-v4-20261117',previous=str(a.previous.resolve()),previous_manifest_sha256=sha(a.previous/'manifest.json'),asset_sha256=previous['asset_sha256'],removed_stable_ids=sorted(removed),new_removals=sorted(NEW_REMOVALS),point_count=runtime,projected_links=a.project_links,
        manual_rules={'14':'-v3_Zc','20':'-v3_Zc','297':'v3_Yc','previous_user_confirmed':'preserved'},implementation_sha256=sha(Path(__file__)),
        generated_sha256={k:sha(a.output/k) for k in ['contact_points.json','contact_frames.json','id_mapping.json']},note='New library only; old generator/checkpoints not changed. Gradient estimates and surface projection need review; no region or dynamics admission.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Complete:',a.output.resolve(),runtime,'points')


if __name__=='__main__':main()
