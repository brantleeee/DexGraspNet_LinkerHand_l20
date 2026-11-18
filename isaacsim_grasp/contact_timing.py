"""Offline contact timing evidence; thresholds never change grasp acceptance."""
import numpy as np


def longest_run(mask):
    padded=np.r_[False,np.asarray(mask,bool),False].astype(int)
    starts=np.flatnonzero(np.diff(padded)==1)
    ends=np.flatnonzero(np.diff(padded)==-1)
    return int(np.max(ends-starts)) if len(starts) else 0


def summarize_contact_timing(trace,links,fingers,case_index,force_threshold=.001,sustain_seconds=.05):
    t=np.asarray(trace['time']);dt=float(trace['dt'])
    if force_threshold<=0 or sustain_seconds<=0:raise ValueError('Positive diagnostic thresholds required')
    if len(t)<2 or not np.allclose(np.diff(t),dt,atol=1e-8):raise ValueError('Uniform physics sampling required')
    start=2.5;gravity=float(trace['gravity_start_seconds']);end=gravity+8*dt
    if t[0]>start or t[-1]<end-1e-8:raise ValueError('Trace does not cover required interval')
    i=case_index;force=trace['finger_normal_force'][:,i];count=trace['finger_contact_count'][:,i]
    lf=trace['link_normal_force_magnitude_sum'][:,i];lc=trace['link_contact_count'][:,i]
    active=(force>force_threshold)&(count>0)
    pre=(t>=start-1e-9)&(t<gravity-1e-9)
    post=(t>=gravity-1e-9)&(t<=end+1e-9)
    finger_ids=[j for j,f in enumerate(fingers) if f!='palm']
    origin=int(np.argmin(abs(t-start)))
    snapshots=[]
    for requested in [2.5,2.7,2.9,gravity,end]:
        k=int(np.argmin(abs(t-requested)))
        snapshots.append(dict(requested_seconds=requested,actual_seconds=float(t[k]),
            fingers={str(n):dict(records=int(count[k,j]),normal_force_n=float(force[k,j])) for j,n in enumerate(fingers)},
            links={str(n):dict(records=int(lc[k,j]),normal_force_n=float(lf[k,j])) for j,n in enumerate(links) if lc[k,j]>0 or lf[k,j]>0},
            displacement_since_action_start_mm=float(np.linalg.norm(trace['object_position'][k,i]-trace['object_position'][origin,i])*1000)))
    durations={str(n):longest_run(active[pre,j])*dt for j,n in enumerate(fingers)}
    had_sustained=any(durations[str(fingers[j])]>=sustain_seconds-1e-9 for j in finger_ids)
    any_finger=active[:,finger_ids].any(axis=1)
    # Each link event is a recorded transition, not proof of physical separation.
    events=[]
    for j,n in enumerate(links):
        for label,mask in [('record',lc[:,j]>0),('load',(lc[:,j]>0)&(lf[:,j]>force_threshold))]:
            for k in np.flatnonzero(mask[1:]!=mask[:-1])+1:
                if start-1e-9<=t[k]<=end+1e-9:
                    events.append(dict(time_seconds=float(t[k]),link=str(n),signal=label,event='established' if mask[k] else 'lost'))
    events.sort(key=lambda e:(e['time_seconds'],e['link'],e['signal']))
    at_gravity=int(np.argmin(abs(t-gravity)))
    if not had_sustained:classification='no_sustained_finger_load_before_gravity'
    elif not any_finger[at_gravity]:classification='preload_finger_load_lost_before_gravity'
    elif not any_finger[post].all():classification='finger_load_lost_after_gravity'
    else:classification='finger_load_persists_in_early_gravity'
    independent=np.asarray(trace['independent_ids'],int)
    movement={}
    for name in ['joint_position','joint_target']:
        q=trace[name][:,i][:,independent]
        movement[name+'_max_change_rad']=float(np.max(abs(q[pre]-q[origin])))
    return dict(classification=classification,snapshots=snapshots,
        preload_max_force_n={str(n):float(force[pre,j].max()) for j,n in enumerate(fingers)},
        preload_longest_loaded_duration_seconds=durations,
        preload_any_finger_max_run_seconds=longest_run(any_finger[pre])*dt,
        preload_peak_displacement_mm=float(np.linalg.norm(trace['object_position'][pre,i]-trace['object_position'][origin,i],axis=-1).max()*1000),
        joint_response=movement,contact_transitions=events,
        preload_load_intervals={str(n):dict(first_seconds=float(t[np.flatnonzero(pre & active[:,j])[0]]) if np.any(pre & active[:,j]) else None,last_seconds=float(t[np.flatnonzero(pre & active[:,j])[-1]]) if np.any(pre & active[:,j]) else None) for j,n in enumerate(fingers)},
        diagnostic_thresholds=dict(force_n=force_threshold,sustained_seconds=sustain_seconds),
        note='Recorded normal-force evidence only; not a force-closure proof or geometry/separation diagnosis. Thresholds do not change strict acceptance.')
