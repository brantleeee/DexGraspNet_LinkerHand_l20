"""Small clipped PPO implementation with timeout bootstrap and reproducible checkpoints."""
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super().__init__()
        def net(out):return nn.Sequential(nn.Linear(obs_dim,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,out))
        self.actor=net(act_dim);self.critic=net(1);self.log_std=nn.Parameter(torch.full((act_dim,),-1.2))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
    def distribution(self,obs):return Normal(self.actor(obs),self.log_std.clamp(-4,0).exp())
    def value(self,obs):return self.critic(obs).squeeze(-1)


def advantages(rewards,values,next_values,done,terminated,gamma=.99,lam=.95):
    result=torch.zeros_like(rewards);gae=torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*next_values[t]*(~terminated[t])-values[t]
        gae=delta+gamma*lam*(~done[t])*gae;result[t]=gae
    return result,result+values


def update(model,optimizer,rollout,epochs=4,mini_batches=4):
    obs,actions,oldlog,oldvalues,returns,adv=rollout
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    n=len(obs);stats=[]
    for _ in range(epochs):
        for idx in torch.randperm(n,device=obs.device).tensor_split(min(mini_batches,n)):
            dist=model.distribution(obs[idx]);logp=dist.log_prob(actions[idx]).sum(-1)
            ratio=(logp-oldlog[idx]).exp();surrogate=torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx])
            value=model.value(obs[idx]);clipped=oldvalues[idx]+(value-oldvalues[idx]).clamp(-.2,.2)
            vloss=torch.maximum((value-returns[idx]).square(),(clipped-returns[idx]).square()).mean()
            entropy=dist.entropy().sum(-1).mean();loss=-surrogate.mean()+.5*vloss-.001*entropy
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            with torch.no_grad():model.log_std.clamp_(-4,0)
            stats.append([float(loss.detach()),float(vloss.detach()),float(entropy.detach())])
    return torch.tensor(stats).mean(0).tolist()
