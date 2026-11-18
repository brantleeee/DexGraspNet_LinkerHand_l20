"""Summarize measured command layers; never infer old raw actions from joint targets."""
import numpy as np


def action_summaries(data):
    gravity=np.asarray(data['time'])>float(data['gravity_start_seconds'])
    n=data['joint_position'].shape[1];result=[]
    for i in range(n):
        row={'gravity_frames':int(gravity.sum()),'enabled_gravity_frames':int(np.asarray(data['action_enabled'])[gravity,i].sum())}
        for key in ['raw_action','requested_residual','effective_requested_residual','applied_residual','joint_velocity']:
            a=np.asarray(data[key])[gravity,i]
            row[key+'_rms']=float(np.sqrt(np.mean(a*a))) if len(a) else 0.
            row[key+'_max_abs']=float(np.max(abs(a))) if len(a) else 0.
            if key!='joint_velocity':
                row[key+'_mean_per_joint']=a.mean(0).tolist() if len(a) else [0.]*data[key].shape[-1]
        a=np.asarray(data['applied_residual'])[gravity,i]
        row['applied_residual_abs_bias_rad']=float(np.mean(abs(a.mean(0)))) if len(a) else 0.
        q=np.asarray(data['joint_position'])[gravity,i]
        row['joint_travel_mean_rad']=float(np.abs(np.diff(q,axis=0)).sum(0).mean()) if len(q)>1 else 0.
        result.append(row)
    return result
