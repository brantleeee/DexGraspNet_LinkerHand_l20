"""Explicit versioned L20 point/frame loading and differentiable normal transforms."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_library(path):
    path=Path(path).resolve();manifest=json.loads((path/'manifest.json').read_text())
    for name,digest in manifest['asset_sha256'].items():
        if sha(name)!=digest:raise ValueError('Contact source asset changed: '+name)
    for name,digest in manifest['generated_sha256'].items():
        if sha(path/name)!=digest:raise ValueError('Contact library file changed: '+name)
    frames=json.loads((path/'contact_frames.json').read_text())
    if [f['runtime_id'] for f in frames]!=list(range(manifest['point_count'])):raise ValueError('Non-dense runtime contact IDs')
    if len({f['stable_id'] for f in frames})!=len(frames):raise ValueError('Duplicate stable contact IDs')
    if set(f['stable_id'] for f in frames)&set(manifest['removed_stable_ids']):raise ValueError('Removed contact still present')
    for f in frames:
        b=np.array([f[k] for k in ['Xc','Yc','Zc']],dtype=float).T
        if not np.isfinite(b).all() or not np.allclose(b.T@b,np.eye(3),atol=1e-5) or np.linalg.det(b)<.999:raise ValueError('Invalid contact frame')
    return manifest,frames


def attach_library(hand,path):
    manifest,frames=load_library(path)
    if hand.n_contact_candidates!=len(frames):raise ValueError('Loaded hand has wrong contact count')
    for i,f in enumerate(frames):
        owner=list(hand.mesh)[int(hand.global_index_to_link_index[i])]
        if owner!=f['parent_link'] or not np.allclose(hand.contact_candidates[i].detach().cpu(),f['origin_P_m'],atol=2e-9,rtol=0):
            raise ValueError('Runtime contact ordering/coordinates differ from library')
    hand.library_normals=torch.tensor([f['Zc'] for f in frames],dtype=hand.contact_candidates.dtype,device=hand.device)
    hand.library_stable_ids=torch.tensor([f['stable_id'] for f in frames],dtype=torch.long,device=hand.device)
    return dict(path=str(Path(path).resolve()),manifest_sha256=sha(Path(path)/'manifest.json'),version=manifest['version'],
        point_count=len(frames),removed_stable_ids=manifest['removed_stable_ids'],
        hand_normal_convention='library Zc in parent link; transformed by FK rotation then wrist rotation; no translation',
        normal_loss='mean((1 + dot(hand_Zc_world, object_outward_normal))**2) over selected contacts',
        original_E_fc_unchanged=True,contact_selection='global random slots, legacy n_contact; no middle participation guarantee')


def world_contact_normals(hand):
    ids=hand.contact_point_indices;local=hand.library_normals[ids]
    owners=hand.global_index_to_link_index[ids];world=torch.zeros_like(local)
    for link,index in hand.link_name_to_link_index.items():
        mask=owners==index
        if mask.any():
            rotated=torch.einsum('bij,bnj->bni',hand.current_status[link].get_matrix()[:,:3,:3],local)
            world=torch.where(mask[...,None],rotated,world)
    return torch.einsum('bij,bnj->bni',hand.global_rotation,world)


def normal_alignment(hand_normal,object_normal):
    h=torch.nn.functional.normalize(hand_normal,dim=-1);o=torch.nn.functional.normalize(object_normal,dim=-1)
    return (1+(h*o).sum(-1).clamp(-1,1)).square().mean(-1)
