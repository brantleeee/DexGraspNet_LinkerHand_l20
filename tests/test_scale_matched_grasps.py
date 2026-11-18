from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'grasp_generation'))
from utils.grasp_results import save_scale_groups, verify_mesh_identity
from utils.joint_constraints import JointConstraints
from utils.object_model import ObjectModel


class ScaleMatchedGraspTests(unittest.TestCase):
    def test_balanced_scales_are_the_rendered_mesh_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'box/coacd/decomposed.obj'
            path.parent.mkdir(parents=True)
            trimesh.creation.box(extents=[1,2,3]).export(path)
            model = ObjectModel(directory, 6, num_samples=0, device='cpu')
            model.initialize('box', scales=[.04,.06])
            scales = model.object_scale_tensor[0].numpy()
            self.assertEqual(np.isclose(scales,.04).sum(),3)
            self.assertEqual(np.isclose(scales,.06).sum(),3)
            for i, scale in enumerate(scales):
                trace = model.get_plotly_data(i)[0]
                np.testing.assert_allclose(np.ptp(np.column_stack((trace.x,trace.y,trace.z)),axis=0),
                                           np.array([1,2,3])*scale,rtol=1e-6)
            with self.assertRaises(ValueError):
                model.initialize('box',scales=[0])

    def test_per_scale_export_preserves_world_pose_without_scaling_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root/'mesh.obj';mesh.write_text('normalized mesh fixture')
            data = [{'scale':s,'qpos':{'WRJTx':float(i),'finger':.2},'source_index':i}
                    for i,s in enumerate([.04,.06,.04,.06])]
            report = save_scale_groups(root,'box',data,np.array([[0,0,0],[1,2,3]]),mesh)
            for group in report['groups']:
                loaded = np.load(root/group['result_file'],allow_pickle=True)
                for record in loaded:
                    self.assertEqual(record, data[record['source_index']])
                    self.assertEqual(record['scale'],group['scale'])
                np.testing.assert_allclose(group['object_extents_m'],np.array([1,2,3])*group['scale'])
                self.assertIsNotNone(verify_mesh_identity(root/group['result_file'],mesh))
            mesh.write_text('accidentally rescaled mesh')
            with self.assertRaises(ValueError):
                verify_mesh_identity(root/report['groups'][0]['result_file'],mesh)

    def test_hard_mimic_limits_and_chain_rule(self):
        constraints = JointConstraints(['drive','dip'],[0,0],[1.5,1.22],
                                        [('dip','drive',1.1619,0)])
        excessive = torch.ones(2,11)*10
        projected = constraints.project(excessive)
        self.assertTrue(torch.all(projected[:,10]<=1.220001))
        torch.testing.assert_close(projected[:,10],1.1619*projected[:,9])
        torch.testing.assert_close(projected[:,:9],excessive[:,:9])
        x = torch.zeros(1,11,dtype=torch.double)
        x[0,9]=.2;x[0,10]=.9;x.requires_grad_()
        p = constraints.project(x)
        (3*p[:,9]**2+7*p[:,10]**2).sum().backward()
        leaf = p.detach().requires_grad_()
        (3*leaf[:,9]**2+7*leaf[:,10]**2).sum().backward()
        pulled = constraints.pullback(leaf.grad)
        torch.testing.assert_close(pulled,x.grad)
        self.assertEqual(pulled[0,10],0.)


if __name__ == '__main__':
    unittest.main()
