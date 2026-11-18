"""Hard URDF limits and mimic projection for projected grasp optimization."""

import torch


class JointConstraints:
    def __init__(self, names, lower, upper, mimics, device='cpu'):
        self.names = list(names)
        self.count = len(names)
        bounds_lo, bounds_hi = list(lower), list(upper)
        remaining = list(mimics)
        self.mimics = []
        pending = {child for child, _, _, _ in remaining}
        while remaining:
            ready = [m for m in remaining if m[1] not in pending]
            if not ready:
                raise ValueError('Cyclic mimic constraints')
            for child, parent, ratio, offset in ready:
                c, p = self.names.index(child), self.names.index(parent)
                self.mimics.append((c,p,float(ratio),float(offset)))
                pending.remove(child)
                remaining.remove((child,parent,ratio,offset))
        for child, parent, ratio, offset in reversed(self.mimics):
            if ratio == 0:
                if not bounds_lo[child] <= offset <= bounds_hi[child]:
                    raise ValueError('Infeasible constant mimic joint')
                continue
            a,b = sorted(((bounds_lo[child]-offset)/ratio, (bounds_hi[child]-offset)/ratio))
            bounds_lo[parent] = max(bounds_lo[parent], a)
            bounds_hi[parent] = min(bounds_hi[parent], b)
        if any(a > b for a,b in zip(bounds_lo,bounds_hi)):
            raise ValueError('Infeasible joint limit intersection')
        self.lower = torch.tensor(bounds_lo, dtype=torch.float, device=device)
        self.upper = torch.tensor(bounds_hi, dtype=torch.float, device=device)
        dependent = {c for c,_,_,_ in self.mimics}
        self.independent_names = [n for i,n in enumerate(self.names) if i not in dependent]

    def project(self, pose):
        if pose.shape[-1] != self.count+9:
            raise ValueError('Pose dimension does not match URDF joints')
        joints = torch.minimum(torch.maximum(pose[...,9:], self.lower.to(pose)), self.upper.to(pose))
        columns = list(joints.unbind(-1))
        for child,parent,ratio,offset in self.mimics:
            columns[child] = ratio*columns[parent]+offset
        return torch.cat((pose[...,:9],torch.stack(columns,dim=-1)),dim=-1)

    def pullback(self, gradient):
        """Chain rule: accumulate dependent DIP gradients into their drives."""
        result = gradient.detach().clone()
        for child,parent,ratio,_ in reversed(self.mimics):
            result[...,9+parent] += ratio*result[...,9+child]
            result[...,9+child] = 0
        return result
