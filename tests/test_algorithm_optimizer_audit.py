import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'grasp_generation'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from utils.optimizer import Annealing
from utils.joint_constraints import JointConstraints
from audit_grasp_algorithm import balance_counterexample


class TinyHand:
    def __init__(self):
        self.device='cpu'
        self.n_dofs=2;self.n_contact_candidates=4
        self.joint_constraints=JointConstraints(['pip','dip'],[0,0],[1.5,1.5],[('dip','pip',.8917,0)])
        self.chain=self
    def forward_kinematics(self,q):return {'tip':q.sin()}
    def set_parameters(self,pose,ids):
        self.hand_pose=pose;self.contact_point_indices=ids
        self.global_translation=pose[:,:3];self.global_rotation=torch.eye(3)[None].repeat(len(pose),1,1)
        self.current_status=self.forward_kinematics(pose[:,9:])
        x=pose[:,9,None].square()+ids*.1+pose[:,0,None]
        y=pose[:,10,None].sin().expand_as(x)
        self.contact_points=torch.stack([x,y,torch.zeros_like(x)],-1)


def test_mixed_accept_reject_restores_pose_contacts_and_correct_gradients():
    torch.manual_seed(7);hand=TinyHand();pose=torch.zeros(2,11);pose[:,9]=torch.tensor([.5,.7]);pose=hand.joint_constraints.project(pose).requires_grad_(True)
    ids=torch.tensor([[0,1],[2,3]]);hand.set_parameters(pose,ids)
    energy=hand.contact_points.square().sum((1,2));energy.sum().backward()
    old_pose=pose.detach().clone();old_ids=ids.clone()
    optimizer=Annealing(hand,device='cpu');optimizer.try_step()
    proposal=hand.hand_pose.detach().clone();proposal_ids=hand.contact_point_indices.clone()
    new_energy=hand.contact_points.square().sum((1,2));new_energy.sum().backward()
    accepted,_=optimizer.accept_step(energy.detach(),energy.detach()+torch.tensor([-1e6,1e6]))
    assert accepted.tolist()==[True,False]
    assert torch.allclose(hand.hand_pose[0],proposal[0]) and torch.equal(hand.contact_point_indices[0],proposal_ids[0])
    assert torch.allclose(hand.hand_pose[1],old_pose[1]) and torch.equal(hand.contact_point_indices[1],old_ids[1])
    fresh=TinyHand();fresh.set_parameters(hand.hand_pose.detach().clone().requires_grad_(True),hand.contact_point_indices.clone())
    fresh.contact_points.square().sum().backward()
    assert torch.allclose(hand.contact_points,fresh.contact_points)
    assert torch.allclose(hand.hand_pose.grad,fresh.hand_pose.grad,atol=1e-6)


def test_zero_balance_score_does_not_certify_wrench_closure():
    example=balance_counterexample()
    assert example['E_fc']==0 and example['normal_ray_wrench_rank']==1


def test_mimic_pullback_matches_independent_directional_derivative():
    c=JointConstraints(['pip','dip'],[0,0],[1.5,1.5],[('dip','pip',.8917,0)])
    q=torch.zeros(1,11);q[0,9]=.4;q=c.project(q).requires_grad_(True)
    loss=(q[0,9]+2*q[0,10]).square();loss.backward();g=c.pullback(q.grad)[0,9]
    values=[]
    for delta in [-.001,.001]:
        v=q.detach().clone();v[0,9]+=delta;v=c.project(v);values.append((v[0,9]+2*v[0,10]).square())
    assert torch.allclose(g,(values[1]-values[0])/.002,atol=.001)


def test_real_l20_mixed_rollback_matches_fresh_fk_normal_and_gradient():
    import numpy as np
    from utils.hand_model import HandModel
    from utils.contact_library import attach_library,world_contact_normals,normal_alignment
    from diagnose_l20_middle_objective import pose_from_record
    root=Path(__file__).resolve().parents[1];library=root/'data/hand_contacts/l20_contact_v6_confirmed_20261117'
    hand=HandModel(model_path=str(root/'grasp_generation/mjcf/linkerhand_l20_right.urdf'),mesh_path=str(root/'grasp_generation/mjcf/meshes'),
        contact_points_path=str(library/'contact_points.json'),penetration_points_path=str(root/'grasp_generation/mjcf/penetration_points.json'),n_surface_points=0,device='cpu')
    attach_library(hand,library)
    data=np.load(root/'data/experiments/l20_v6_roots12_drill_20261117/results/scales/scale_1/power-drill-clean-l20.npy',allow_pickle=True)
    pose=torch.cat([pose_from_record(data[i],hand,'cpu')[0] for i in [21,7]]).requires_grad_(True)
    ids=torch.tensor([data[i]['contact_point_indices'] for i in [21,7]])
    def loss():
        n=world_contact_normals(hand)
        return hand.contact_points.square().sum((1,2))+normal_alignment(n,torch.tensor([[[0.,0.,1.]]]).expand_as(n))
    hand.set_parameters(pose,ids);old=loss();old.sum().backward()
    optimizer=Annealing(hand,device='cpu');optimizer.try_step();loss().sum().backward()
    accepted,_=optimizer.accept_step(old.detach(),old.detach()+torch.tensor([-1e6,1e6]));assert accepted.tolist()==[True,False]
    stored_grad=hand.hand_pose.grad.clone();stored_points=hand.contact_points.detach().clone();stored_normals=world_contact_normals(hand).detach().clone()
    hand.set_parameters(hand.hand_pose.detach().clone().requires_grad_(True),hand.contact_point_indices.clone());loss().sum().backward()
    assert torch.allclose(stored_points,hand.contact_points,atol=1e-6)
    assert torch.allclose(stored_normals,world_contact_normals(hand),atol=1e-6)
    assert torch.allclose(stored_grad,hand.hand_pose.grad,atol=2e-5,rtol=1e-4)
