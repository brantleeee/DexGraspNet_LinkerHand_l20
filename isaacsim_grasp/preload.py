"""Pure phase schedule for the v5 settle-tail intervention experiment."""
import math
import numpy as np


def action_start_step(close, settle, preload_seconds, dt=1/240, decimation=4):
    seconds=float(preload_seconds)
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError('preload-seconds must be finite and positive')
    steps=round(seconds/dt)
    if abs(steps*dt-seconds)>1e-9 or steps>settle or steps<decimation:
        raise ValueError('Preload must fit entirely within the zero-gravity settle phase')
    start=close+settle-steps
    if any(s%decimation for s in (close,close+settle,start)):
        raise ValueError('Preload and phase boundaries must align with control ticks')
    return start


def eligible(step, start, total, failed):
    return (step>=start)&(step<total)&(~np.asarray(failed,dtype=bool))


def response_summary(data, preload_seconds, mimics=()):
    """Per-environment diagnostics at the loading boundary; no success relabeling."""
    t=np.asarray(data['time']);gravity=float(data['gravity_start_seconds'])
    start=gravity-float(preload_seconds);window=(t>start-1e-8)&(t<=gravity+1e-8)
    if not window.any():return []
    boundary=int(np.argmin(abs(t-gravity)))
    if abs(float(t[boundary])-gravity)>1e-8:return []
    initial=int(np.argmin(abs(t-start)))
    q=np.asarray(data['joint_position']);ids=np.asarray(data['independent_ids'],int)
    result=[]
    for i in range(q.shape[1]):
        result.append(dict(env=i,preload_start_seconds=start,gravity_start_seconds=gravity,
            force_at_gravity_n=float(data['contact_force'][boundary,i]),
            force_max_preload_n=float(np.max(data['contact_force'][window,i])),
            penetration_max_preload_m=float(np.max(data['penetration'][window,i])),
            requested_residual_max_preload_rad=float(np.max(abs(data['requested_residual'][window,i]))),
            applied_residual_max_preload_rad=float(np.max(abs(data['applied_residual'][window,i]))),
            active_joint_motion_max_preload_rad=float(np.max(abs(q[window,i][:,ids]-q[initial,i,ids]))),
            object_translation_preload_m=float(np.linalg.norm(data['object_position'][boundary,i]-data['object_position'][initial,i])),
            mimic_error_max_preload_rad=max([float(np.max(abs(q[window,i,c]-m*q[window,i,p]-b))) for c,p,m,b in mimics] or [0.]),
            note='Contact force is a sensor aggregate; nonzero force alone is not stable grasp evidence'))
    return result
