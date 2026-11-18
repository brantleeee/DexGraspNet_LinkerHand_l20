"""Evaluation-only hold of applied offsets; leaves original controller source intact."""
import numpy as np


class AppliedOffsetHold:
    def __init__(self,controller,clock,start_step,end_step):
        if not 0<=start_step<end_step:raise ValueError('Invalid hold interval')
        self.controller=controller;self.clock=clock;self.start=start_step;self.end=end_step
        self.original_step=controller.step;self.snapshot=None;self.active=False
        controller.step=self.step

    def step(self,reference,action,dt,gate=1.):
        k=self.clock();self.active=self.start<=k<self.end
        if not self.active:return self.original_step(reference,action,dt,gate)
        if self.snapshot is None:self.snapshot=self.controller.offset.copy()
        target=reference.copy();target[:,self.controller.independent]+=self.snapshot
        projected=self.controller.project(target)
        self.controller.offset=projected[:,self.controller.independent]-reference[:,self.controller.independent]
        if not np.allclose(self.controller.offset,self.snapshot,atol=1e-6,rtol=0):
            raise RuntimeError('Reference changed during hold; fixed settle reference required')
        return projected
