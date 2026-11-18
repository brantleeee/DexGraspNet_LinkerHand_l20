"""Run in the local GraspQP environment; verify removal of hidden joint bias."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from graspqp_drill_pipeline import exact_runtime_kinematics


def test_exact_angles_and_finite_zero_gradient():
    from pytorch_kinematics import frame
    exact_runtime_kinematics()
    f=SimpleNamespace(joint=SimpleNamespace(axis=torch.tensor([0.,0.,1.]),joint_type='revolute',offset=frame.tf.Transform3d()))
    theta=torch.tensor([[0.],[.2],[-.2]],requires_grad=True)
    mat=frame.Frame.get_transform(f,theta).get_matrix()
    expected=Rotation.from_rotvec(np.array([[0,0,0],[0,0,.2],[0,0,-.2]])).as_matrix()
    assert np.allclose(mat.detach().numpy()[:,:3,:3],expected,atol=1e-6)
    mat.sum().backward()
    assert torch.isfinite(theta.grad).all()
