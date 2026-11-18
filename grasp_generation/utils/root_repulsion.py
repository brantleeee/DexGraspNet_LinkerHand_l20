"""Opt-in root assembly exclusions for legacy keypoint self repulsion."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from l20_root_collision_policy import IGNORED,THUMB_ROOTS


def ignored_pairs(policy):
    if policy=='all':return set()
    if policy=='angle-only':return IGNORED
    if policy=='all-roots-angle-only':return IGNORED|THUMB_ROOTS
    raise ValueError('Unknown root contact policy')


def allowed_mask(owners,policy,device):
    excluded=ignored_pairs(policy)
    return torch.tensor([[frozenset((a,b)) not in excluded for b in owners] for a in owners],dtype=torch.bool,device=device)


def filtered_self_repulsion(hand,policy):
    if policy=='all':return hand.self_penetration()
    key='_root_repulsion_mask_'+policy
    if not hasattr(hand,key):
        owners=[list(hand.mesh)[i] for i in hand.global_index_to_link_index_penetration.tolist()]
        setattr(hand,key,allowed_mask(owners,policy,hand.device))
    points=hand.get_penetraion_keypoints()
    distance=(points[:,:,None]-points[:,None,:]+1e-13).square().sum(-1).sqrt()
    distance=torch.where(distance<1e-6,torch.full_like(distance,1e6),distance)
    return ((.02-distance).clamp_min(0)*getattr(hand,key)).sum(dim=(-1,-2))
