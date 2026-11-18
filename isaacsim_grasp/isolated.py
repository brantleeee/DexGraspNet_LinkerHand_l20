"""CPU utilities for fresh-process grasp validation and provenance checks."""
import json
import numpy as np
from .paired_evaluation import validate_samples


def select_group(samples, group):
    offsets=validate_samples(samples,samples['episodes'],samples['num_envs'],samples['amplitude_m'])
    groups=offsets.reshape(-1,6,3)
    if group<0 or group>=len(groups):raise ValueError('Group outside saved sample deck')
    result=dict(samples,episodes=1,num_envs=6,offsets_m=groups[group:group+1].tolist(),
                source_group=group)
    validate_samples(result,1,6,samples['amplitude_m'])
    return result


def compare_traces(a,b,prefix_seconds=2.5,position_tolerance_m=1e-5,joint_tolerance_rad=1e-4):
    """Tolerances are diagnostic only, never modifications to grasp acceptance."""
    if not np.array_equal(a['time'],b['time']):raise ValueError('Trace sampling differs')
    mask=np.asarray(a['time'])<=prefix_seconds+1e-9
    if not mask.any():raise ValueError('No common prefix')
    def maxabs(key):return float(np.max(abs(a[key][mask]-b[key][mask])))
    position=np.linalg.norm(a['object_position'][mask]-b['object_position'][mask],axis=-1)
    joint=maxabs('joint_position');target=maxabs('joint_target')
    angle=np.degrees(2*np.arccos(np.clip(abs(np.sum(a['object_quaternion'][mask]*b['object_quaternion'][mask],axis=-1)),0,1)))
    # Normalize before quaternion comparisons; sensor quaternions may be float32.
    qa=np.asarray(a['object_quaternion'][mask],float);qb=np.asarray(b['object_quaternion'][mask],float)
    qa/=np.linalg.norm(qa,axis=-1,keepdims=True);qb/=np.linalg.norm(qb,axis=-1,keepdims=True)
    angle=np.degrees(2*np.arccos(np.clip(abs(np.sum(qa*qb,axis=-1)),0,1)))
    fields=dict(max_position_difference_m=float(position.max()),max_joint_difference_rad=joint,
        max_target_difference_rad=target,max_angle_difference_deg=float(angle.max()),
        max_force_difference_n=maxabs('contact_force'),max_velocity_difference_m_s=maxabs('object_linear_velocity'))
    passed=(fields['max_position_difference_m']<=position_tolerance_m and joint<=joint_tolerance_rad and target<=1e-6
            and fields['max_angle_difference_deg']<=.01 and fields['max_velocity_difference_m_s']<=1e-3
            and fields['max_force_difference_n']<=.05)
    return dict(prefix_seconds=prefix_seconds,passed=bool(passed),**fields,
        tolerances=dict(position_m=position_tolerance_m,joint_rad=joint_tolerance_rad,target_rad=1e-6,
                        angle_deg=.01,linear_velocity_m_s=1e-3,force_n=.05),
        note='Numerical process-repeatability thresholds, separate from physical grasp success.')


def finger_name(link):
    for finger in ['thumb','index','middle','ring','pinky']:
        if link.startswith(finger+'_'):return finger
    return 'palm'


def contact_events(times,active,minimum_gap_frames=12):
    """Summarize establishment and >=50ms gaps at 240Hz without relabeling success."""
    times=np.asarray(times);active=np.asarray(active,bool)
    touched=np.flatnonzero(active);events=[]
    changes=np.flatnonzero(active[1:]!=active[:-1])+1
    for k in changes:
        if active[k]:events.append(dict(time_seconds=float(times[k]),event='contact_established'))
        else:
            end=k
            while end<len(active) and not active[end]:end+=1
            if end-k>=minimum_gap_frames:
                events.append(dict(time_seconds=float(times[k]),event='contact_gap',duration_seconds=float(times[end-1]-times[k]+(times[1]-times[0]))))
    return dict(first_contact_seconds=float(times[touched[0]]) if len(touched) else None,
                last_contact_seconds=float(times[touched[-1]]) if len(touched) else None,events=events)
