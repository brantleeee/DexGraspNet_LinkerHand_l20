"""Read-only first-failure tracking and admission checks for gravity-only training."""
import json
from pathlib import Path
import numpy as np
from .history import REASONS


class FirstFailureRecorder:
    def __init__(self,history):
        self.history=history;self.begin();original=history.record
        def record(*args,**kwargs):
            before=history.failed.copy();original(*args,**kwargs)
            for i in np.flatnonzero(history.failed&~before):
                step=history.step
                phase=('acquisition_boundary' if step==history.start and history.reasons[i,3] else
                       'close_zero_g' if step<=history.close else 'settle_zero_g' if step<=history.start else 'gravity_test')
                self.first[i]=dict(physics_step=step,time_seconds=step*history.dt,phase=phase,
                    gravity_seconds=max(0.,(step-history.start)*history.dt),
                    reasons=[name for k,name in enumerate(REASONS) if history.reasons[i,k]])
        history.record=record
    def begin(self):self.first=[None for _ in range(self.history.n)]


def admission(rows):
    if not rows:raise ValueError('No baseline trials')
    prep=[];gravity=[];passed=[]
    for r in rows:
        fail=r['first_failure']
        if bool(r['passed'])!=(fail is None):raise ValueError('Baseline outcome/first-failure mismatch')
        if r['passed']:passed.append(r)
        elif fail['phase']=='gravity_test':gravity.append(r)
        else:prep.append(r)
    suitable=bool(gravity) and not prep and all(r['acquired'] for r in rows)
    reason='gravity_failures_after_valid_preparation' if suitable else 'preparation_failure_not_repairable_by_gravity_only' if prep else 'no_baseline_failure_to_repair'
    times=[r['first_failure']['gravity_seconds'] for r in gravity if 'gravity_seconds' in r['first_failure']]
    return dict(suitable=suitable,reason=reason,trials=len(rows),passed=len(passed),gravity_failures=len(gravity),preparation_failures=len(prep),
                earliest_gravity_failure_seconds=min(times) if times else None,
                note='Admission concerns observed baseline samples only, not general robustness or guaranteed RL repair.')


def validate_baseline(folder,reference,config,samples):
    folder=Path(folder);data=json.loads((folder/'baseline.json').read_text())
    if not data['complete']:raise ValueError('Baseline incomplete')
    old=json.loads((folder/'reference_snapshot.json').read_text())
    for k in ['object','candidate','scale','scene_sha256','source_sha256','targets']:
        if reference[k]!=old[k]:raise ValueError(f'Baseline reference differs: {k}')
    saved=json.loads((folder/'config.json').read_text())
    for k in ['position_noise','residual_scale','residual_rate','kp','kd','physics_dt','decimation','close_seconds','settle_seconds','gravity_seconds','open_fraction','collision','mimic']:
        if config[k]!=saved[k]:raise ValueError(f'Baseline physical/control setting differs: {k}')
    previous=json.loads((folder/'perturbations.json').read_text())
    if samples['offsets_m']!=previous['offsets_m']:raise ValueError('Baseline and evaluation samples differ')
    return data
