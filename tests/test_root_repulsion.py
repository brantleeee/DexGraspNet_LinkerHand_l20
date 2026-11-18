import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'grasp_generation'))
from utils.root_repulsion import allowed_mask,ignored_pairs


def test_root_mask_removes_only_twelve_unordered_link_pairs():
    owners=['base_link']+[f+'_'+x for f in ['index','middle','ring','pinky'] for x in ['metacarpals','proximal','middle','distal']]+['thumb_metacarpals_base1','thumb_metacarpals_base2','thumb_metacarpals','thumb_proximal','thumb_distal']
    mask=allowed_mask(owners,'all-roots-angle-only','cpu')
    assert (~mask).sum()==24 and torch.equal(mask,mask.T)
    assert mask[owners.index('middle_proximal'),owners.index('index_proximal')]
    assert mask[owners.index('thumb_distal'),owners.index('base_link')]
    assert len(ignored_pairs('all-roots-angle-only'))==12


def test_excluded_root_penalty_has_zero_gradient_but_cross_finger_remains():
    q=torch.tensor(.005,requires_grad=True)
    points=torch.stack([torch.stack([q,q*0,q*0]),torch.zeros(3),torch.tensor([.015,0,0])])
    dist=(points[:,None]-points[None,:]+1e-13).square().sum(-1).sqrt()
    mask=allowed_mask(['middle_proximal','base_link','index_proximal'],'all-roots-angle-only','cpu')
    root=(.02-dist[0,1]).clamp_min(0)*mask[0,1];cross=(.02-dist[0,2]).clamp_min(0)*mask[0,2]
    assert torch.autograd.grad(root,q,retain_graph=True)[0]==0
    assert torch.autograd.grad(cross,q)[0].abs()>.9
