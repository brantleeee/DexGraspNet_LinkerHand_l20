"""Finger-stratified contact slots shared by initialization and proposal sampling."""
import torch


def slot_groups(mode):
    if mode=='global':return None
    if mode=='grouped4':return [('thumb',),('index',),('middle',),('ring','pinky')]
    if mode=='grouped5':return [(f,) for f in ['thumb','index','middle','ring','pinky']]
    raise ValueError('Unknown contact sampling mode')


def sample_contacts(hand,batch_size,n_contact,previous=None,switch_mask=None):
    groups=slot_groups(getattr(hand,'contact_sampling','global'))
    if groups is None:
        if previous is None:return torch.randint(hand.n_contact_candidates,(batch_size,n_contact),device=hand.device)
        result=previous.clone();result[switch_mask]=torch.randint(hand.n_contact_candidates,(int(switch_mask.sum()),),device=hand.device);return result
    if len(groups)!=n_contact:raise ValueError('Contact slot count disagrees with grouping')
    cache='_contact_pools_'+hand.contact_sampling
    if not hasattr(hand,cache):
        owners=[list(hand.mesh)[i].split('_')[0] for i in hand.global_index_to_link_index.tolist()]
        pools=[torch.tensor([i for i,f in enumerate(owners) if f in group],device=hand.device,dtype=torch.long) for group in groups]
        if any(not len(pool) for pool in pools):raise ValueError('Empty finger contact pool')
        setattr(hand,cache,pools)
    result=torch.empty((batch_size,n_contact),dtype=torch.long,device=hand.device) if previous is None else previous.clone()
    for j,group in enumerate(groups):
        pool=getattr(hand,cache)[j]
        if not len(pool):raise ValueError('Empty finger group: '+str(group))
        if previous is not None and not torch.isin(previous[:,j],pool).all():raise ValueError('Previous contacts violate group')
        draw=pool[torch.randint(len(pool),(batch_size,),device=hand.device)]
        result[:,j]=draw if previous is None else torch.where(switch_mask[:,j],draw,previous[:,j])
    if (torch.sort(result,dim=1).values.diff(dim=1)==0).any():raise ValueError('Duplicate grouped contact ID')
    return result
