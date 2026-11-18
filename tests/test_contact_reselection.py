from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reselect_l20_middle_contacts import balance_terms,choose
from l20_objective_diagnostics import contact_terms


def test_balance_matches_generation_algebra():
    points=torch.tensor([[[.1,.2,.3],[.3,-.2,.2],[-.1,.3,.4],[0.,-.1,.1]]])
    normals=torch.tensor([[[1.,0,0],[0,1,0],[0,0,1],[-1.,0,0]]])
    class Object:
        def cal_distance(self,p):return torch.ones(1,4),normals
    original,*_=contact_terms(points,Object())
    force,moment=balance_terms(points,normals)
    assert torch.allclose(original,force+moment,atol=1e-6)
    assert torch.allclose(force,torch.tensor([2.]))


def test_reselection_never_admits_rejected_geometry():
    bad=dict(admissible=False,total=-100.,row_index=1)
    good=dict(admissible=True,total=2.,row_index=2)
    unknown=dict(admissible=True,total=None,row_index=3)
    assert choose([bad,good,unknown],'total') is good
    assert choose([bad,unknown],'total') is None
