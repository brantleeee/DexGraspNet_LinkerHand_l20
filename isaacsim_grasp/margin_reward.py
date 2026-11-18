"""v3 continuous stability-margin objective, separate from unchanged strict acceptance."""
import numpy as np

DEFAULT_WEIGHTS=dict(position_scale_m=.001,angle_scale_deg=2.,velocity_scale_m_s=.02,spin_scale_rad_s=.3,
    pose_weight=3.,contact_weight=1.,residual_scale_rad=.05,applied_residual_weight=.3,requested_residual_weight=.05,
    residual_rate_weight=.05,joint_velocity_weight=.03,penetration_weight=2.,history_slip_weight=.5,history_angle_weight=.25)


def margin_terms(position,angle,speed,spin,force,depth,offset,requested,previous_offset,qd,max_slip,max_angle,control_dt,rate,weights=None):
    w=dict(DEFAULT_WEIGHTS,**(weights or {}))
    pose=np.exp(-np.minimum((position/w['position_scale_m'])**2+(angle/w['angle_scale_deg'])**2,40))
    motion=np.exp(-np.minimum((speed/w['velocity_scale_m_s'])**2+(spin/w['spin_scale_rad_s'])**2,40))
    terms=dict(positive=(w['pose_weight']+w['contact_weight']*(force>.001))*pose*motion,
        applied_residual=w['applied_residual_weight']*np.mean((offset/w['residual_scale_rad'])**2,axis=1),
        requested_residual=w['requested_residual_weight']*np.mean((requested/w['residual_scale_rad'])**2,axis=1),
        residual_rate=w['residual_rate_weight']*np.mean(((offset-previous_offset)/(rate*control_dt))**2,axis=1),
        joint_velocity=w['joint_velocity_weight']*np.mean(np.minimum(qd**2,25),axis=1),
        penetration=w['penetration_weight']*np.minimum(depth/.001,5),
        history_slip=w['history_slip_weight']*np.minimum((max_slip/.005)**2,25),
        history_angle=w['history_angle_weight']*np.minimum((max_angle/15)**2,25))
    terms['costs']=sum(v for k,v in terms.items() if k!='positive')
    return terms
