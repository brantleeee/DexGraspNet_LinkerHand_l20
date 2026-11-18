import sys,types
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'grasp_generation'))
from utils.contact_sampling import sample_contacts


def test_grouped_sampling_keeps_all_fingers_and_unique_ids_over_switches():
    names=['thumb_distal','index_distal','middle_distal','ring_distal','pinky_distal']
    hand=types.SimpleNamespace(device='cpu',mesh=dict.fromkeys(names),global_index_to_link_index=torch.arange(5).repeat_interleave(3),n_contact_candidates=15,contact_sampling='grouped5')
    ids=sample_contacts(hand,32,5)
    for _ in range(40):
        old=ids.clone();mask=torch.rand(32,5)<.5;ids=sample_contacts(hand,32,5,previous=ids,switch_mask=mask)
        assert torch.equal(ids[~mask],old[~mask])
        assert torch.equal(hand.global_index_to_link_index[ids],torch.arange(5).expand(32,5))
        assert not (ids.sort(1).values.diff(dim=1)==0).any()


def test_raw_hand_queries_the_original_triangles_instead_of_convex_hull():
    from utils.hand_model_v2 import HandModel
    from torchsdf import index_vertices_by_faces
    hand=HandModel(model_path=str(ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf'),mesh_path=str(ROOT/'grasp_generation/mjcf/meshes'),contact_points_path=str(ROOT/'data/hand_contacts/l20_contact_v6_confirmed_20261117/contact_points.json'),n_surface_points=0,device='cpu',force_mesh_sdf=True)
    assert hand.penetration_geometry=='raw-mesh'
    for m in hand.mesh.values():assert torch.equal(m['face_verts'],index_vertices_by_faces(m['vertices'],m['faces'].long()))
    middle=next(x for x in hand.geometry_audit if x['link']=='middle_middle')
    assert middle['query_faces']==middle['raw_faces'] and middle['query_faces']!=middle['legacy_hull_faces']
