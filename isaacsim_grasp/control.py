"""Pure numpy joint target projection and bounded/rate-limited residual control."""
import numpy as np


class ResidualTargets:
    def __init__(self,joints,names,amplitude=.15,rate=.2):
        self.names=list(names);self.amplitude=amplitude;self.rate=rate
        lookup={j['name']:j for j in joints};self.joints=[lookup[n] for n in names]
        self.independent=[i for i,j in enumerate(self.joints) if not j['mimic']]
        self.lower=np.array([j['limit']['lower'] for j in self.joints]);self.upper=np.array([j['limit']['upper'] for j in self.joints])
        self.mimics=[];pending=[j for j in joints if j['mimic']]
        while pending:
            ready=[j for j in pending if j['mimic']['joint'] not in {p['name'] for p in pending}]
            if not ready:raise ValueError('Cyclic mimic joints')
            for j in ready:
                m=j['mimic'];c=self.names.index(j['name']);p=self.names.index(m['joint'])
                self.mimics.append((c,p,float(m.get('multiplier',1)),float(m.get('offset',0))));pending.remove(j)
        for c,p,m,b in reversed(self.mimics):
            if m==0:
                if not self.lower[c]<=b<=self.upper[c]:raise ValueError('Infeasible mimic')
            else:
                low,high=sorted([(self.lower[c]-b)/m,(self.upper[c]-b)/m]);self.lower[p]=max(self.lower[p],low);self.upper[p]=min(self.upper[p],high)
        if np.any(self.lower>self.upper):raise ValueError('Empty joint range')
        # Rate is enforced for both independent and induced mimic target offsets.
        self.rates=np.full(len(names),rate)
        for c,p,m,b in reversed(self.mimics):
            if m:self.rates[p]=min(self.rates[p],self.rates[c]/abs(m))
        self.offset=None

    def reset(self,n):self.offset=np.zeros((n,len(self.independent)),dtype=np.float32)

    def project(self,q):
        q=np.clip(q,self.lower,self.upper)
        for c,p,m,b in self.mimics:q[:,c]=m*q[:,p]+b
        return q.astype(np.float32)

    def step(self,reference,action,dt,gate=1.):
        if self.offset is None:self.reset(len(reference))
        desired=self.amplitude*np.tanh(action)*gate
        delta=self.rates[self.independent]*dt
        self.offset+=np.clip(desired-self.offset,-delta,delta)
        target=reference.copy();target[:,self.independent]+=self.offset
        projected=self.project(target)
        self.offset=projected[:,self.independent]-reference[:,self.independent]
        return projected
