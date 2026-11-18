"""Contact coverage, proposal invariants, and fail-fast checks."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grasp_generation'))
from utils.contact_sampling import sample_contacts


def hand_fixture():
    # Deliberately shuffled link order: slots must follow finger ownership,
    # not assume contact IDs are stored in thumb-to-pinky order.
    names=['middle_distal','pinky_distal','index_middle','thumb_distal','ring_distal']
    return SimpleNamespace(device='cpu',mesh=dict.fromkeys(names),
        global_index_to_link_index=torch.arange(5).repeat_interleave(3),
        n_contact_candidates=15,contact_sampling='grouped5')


def test_initial_and_repeated_proposals_preserve_fingers_and_unswitched_slots():
    torch.manual_seed(31)
    hand=hand_fixture()
    ids=sample_contacts(hand,32,5)
    for step in range(42):
        mask=(torch.zeros(32,5,dtype=torch.bool) if step==0 else
              torch.ones(32,5,dtype=torch.bool) if step==1 else torch.rand(32,5)<.5)
        old=ids.clone()
        ids=sample_contacts(hand,32,5,previous=old,switch_mask=mask)
        assert torch.equal(ids[~mask],old[~mask])
        assert all(len(set(row))==5 for row in ids.tolist())
        owners=[[list(hand.mesh)[i].split('_')[0] for i in row]
                for row in hand.global_index_to_link_index[ids].tolist()]
        assert owners==[['thumb','index','middle','ring','pinky']]*32


def test_invalid_slot_count_and_missing_finger_are_rejected():
    with pytest.raises(ValueError,match='slot count'):
        sample_contacts(hand_fixture(),2,4)
    hand=hand_fixture()
    hand.global_index_to_link_index[:3]=1  # no middle candidates
    with pytest.raises(ValueError,match='Empty finger'):
        sample_contacts(hand,2,5)


def test_invalid_existing_assignment_is_not_silently_reused():
    hand=hand_fixture()
    ids=sample_contacts(hand,2,5)
    ids[:,2]=ids[:,0]
    with pytest.raises(ValueError,match='Previous contacts'):
        sample_contacts(hand,2,5,previous=ids,switch_mask=torch.zeros_like(ids,dtype=torch.bool))
