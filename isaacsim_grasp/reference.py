"""Load one exact generated candidate without rescaling or silent projection."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(results, obj, candidate):
    root=Path(results).resolve()
    batch=json.loads((root/obj/'summary.json').read_text())
    matches=[r for r in batch if r['candidate']==candidate]
    if len(matches)!=1 or not matches[0]['completed']:raise ValueError('Select one completed original candidate')
    folder=Path(matches[0]['output'])
    summary=json.loads((folder/'summary.json').read_text());m=summary['manifest']
    c=next(c for c in m['candidates'] if c['index']==candidate)
    if not c['feasibility']['source_pose_feasible']:raise ValueError('Original reference violates URDF constraints')
    if digest(m['source'])!=m['source_sha256'] or digest(m['urdf'])!=m['urdf_sha256']:raise ValueError('Original source/URDF changed')
    for mesh in m['meshes']:
        if digest(mesh['path'])!=mesh['sha256']:raise ValueError('Original mesh changed')
    audit=json.loads((folder/'collision_audit.json').read_text())
    if not audit['all_sdf'] or audit['active_count']!=23:raise ValueError('Reference is not the audited SDF model')
    drives=json.loads((folder/'runtime_drives.json').read_text())
    if sorted(v for v in drives['kp'][0] if v)!=[5.]*16:raise ValueError('Reference uses unexpected PD gains')
    if len([v for v in drives['kd'][0] if v])!=16 or any(abs(v-.05)>1e-5 for v in drives['kd'][0] if v):raise ValueError('Reference damping differs')
    if summary['parameters']['sdf_resolution']!=256 or summary['parameters']['friction']!=3:raise ValueError('Expected SDF256 and friction3 reference')
    joints=[j for j in m['joints'] if j['type']!='fixed']
    return dict(schema=1,object=obj,candidate=candidate,scale=c['scale'],qpos=c['qpos'],targets=c['targets'],
        joints=joints,source_sha256=m['source_sha256'],urdf_sha256=m['urdf_sha256'],
        scene=str((folder/'scene.usda').resolve()),scene_sha256=digest(folder/'scene.usda'),
        hand_links=[link['name'] for link in m['links']],validation_parameters=summary['parameters'],
        source_results=str(root),reference_is_successful=summary['all_directions_pass'])
