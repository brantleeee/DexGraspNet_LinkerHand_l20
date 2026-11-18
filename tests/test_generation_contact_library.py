from pathlib import Path
import sys,types
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'grasp_generation'))
from utils.contact_library import world_contact_normals,normal_alignment


def test_antiparallel_normals_have_zero_cost_but_parallel_normals_do_not():
    h=torch.tensor([[[1.,0,0]]])
    assert normal_alignment(h,-h).item()==0
    assert normal_alignment(h,h).item()==4


def test_fk_rotates_normals_without_translation_and_preserves_gradients():
    angle=torch.tensor(.4,requires_grad=True);z=torch.zeros_like(angle);one=torch.ones_like(angle)
    rot=torch.stack([torch.stack([angle.cos(),-angle.sin(),z]),torch.stack([angle.sin(),angle.cos(),z]),torch.stack([z,z,one])])
    matrix=torch.cat([torch.cat([rot,torch.tensor([[10.],[20.],[30.]])],1),torch.tensor([[0.,0.,0.,1.]])],0)[None]
    hand=types.SimpleNamespace(contact_point_indices=torch.tensor([[0]]),library_normals=torch.tensor([[1.,0,0]]),
        global_index_to_link_index=torch.tensor([0]),link_name_to_link_index={'finger':0},
        current_status={'finger':types.SimpleNamespace(get_matrix=lambda:matrix)},global_rotation=torch.eye(3)[None])
    n=world_contact_normals(hand)
    assert torch.allclose(n[0,0],torch.stack([angle.cos(),angle.sin(),z]))
    cost=normal_alignment(n,torch.tensor([[[0.,1.,0.]]]))
    cost.backward();assert angle.grad is not None and abs(angle.grad.item())>.1
