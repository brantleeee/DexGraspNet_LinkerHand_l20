"""Replay saved 240Hz trajectories through v3/v4 reward formulas without physics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.history import EpisodeHistory
from isaacsim_grasp.margin_reward import margin_terms
from isaacsim_grasp.environment_v4 import nonnegative_margin


def audit(results,output):
    root=Path(results);output=Path(output)
    if output.exists():raise ValueError('Output must be new')
    summary=json.loads((root/'evaluation.json').read_text());cfg=json.loads((root/'config.json').read_text())
    if not summary['complete']:raise ValueError('Evaluation is incomplete')
    trials=[]
    for mode in ['zero_residual','policy']:
        for episode,path in enumerate(summary['diagnostics'][mode]):
            with np.load(root/path) as archive:z={k:archive[k] for k in archive.files}
            n=z['object_position'].shape[1];dt=float(z['dt']);start=round(float(z['gravity_start_seconds'])/dt)
            end=len(z['time'])-1;close=round(float(z['close_seconds'])/dt)
            if not np.allclose(np.diff(z['time']),dt,atol=1e-8,rtol=0):raise ValueError('Missing physics frames')
            source=[r for r in summary['runs'][mode] if r['episode']==episode]
            source=sorted(source,key=lambda r:r['env'])
            if len(source)!=n:raise ValueError('Outcome/trace coverage differs')
            h=EpisodeHistory(n,close,start-close,end-start,dt)
            sums=np.zeros((2,n));discounted=np.zeros((2,n));alive_sum=np.zeros((2,n));alive_n=np.zeros(n,int)
            for t in range(1,end+1):
                if (t-1)%4==0:before=h.failed.copy()
                # Acquisition uses the actual saved acquisition flags, not an
                # inferred contact-force threshold. Counts otherwise do not affect reward.
                count=np.array([int(r['acquired']) for r in source]) if t==start else np.zeros(n,int)
                h.record(t,z['penetration'][t],count,z['contact_force'][t],z['object_position'][t],z['object_quaternion'][t])
                if t%4:continue
                pos=np.linalg.norm(z['object_position'][t]-h.ref_pos,axis=1)
                angle=np.degrees(2*np.arccos(np.clip(abs((z['object_quaternion'][t]*h.ref_quat).sum(1)),0,1)))
                terms=margin_terms(pos,angle,np.linalg.norm(z['object_linear_velocity'][t],axis=1),
                    np.linalg.norm(z['object_angular_velocity'][t],axis=1),z['contact_force'][t],z['penetration'][t],
                    z['applied_residual'][t],z['effective_requested_residual'][t],z['applied_residual'][t-4],
                    z['joint_velocity'][t],h.max_slip,h.max_angle,4*dt,cfg['residual_rate'],cfg['margin_reward'])
                rewards=np.stack([h.reward(terms['positive'],terms['costs'],before),
                    h.reward(nonnegative_margin(terms['positive'],terms['costs']),np.zeros(n),before)])
                sums+=rewards
                if t>start:
                    discounted+=.99**((t-start)//4-1)*rewards
                    alive=~h.failed;alive_n+=alive;alive_sum+=np.where(alive[None,:],rewards,0)
            if h.passed().tolist()!=[r['passed'] for r in source]:raise ValueError('Replay strict outcomes differ from recorded outcomes')
            for i,r in enumerate(source):
                trials.append(dict(control=mode,episode=episode,env=i,case=r['case'],recorded_passed=r['passed'],
                    v3_total=float(sums[0,i]),v4_replayed_total=float(sums[1,i]),
                    v3_discounted=float(discounted[0,i]),v4_replayed_discounted=float(discounted[1,i]),
                    v3_live_mean=float(alive_sum[0,i]/max(alive_n[i],1)),v4_replayed_live_mean=float(alive_sum[1,i]/max(alive_n[i],1))))
    stats={}
    for mode in ['zero_residual','policy']:
        rows=[r for r in trials if r['control']==mode]
        stats[mode]=dict(trials=len(rows),recorded_passed=sum(r['recorded_passed'] for r in rows),
            v3_negative_live_trials=sum(r['v3_live_mean']<0 for r in rows),v4_negative_live_trials=sum(r['v4_replayed_live_mean']<0 for r in rows),
            v3_discounted_min=min(r['v3_discounted'] for r in rows),v4_discounted_min=min(r['v4_replayed_discounted'] for r in rows))
    output.mkdir(parents=True)
    result=dict(source=str(root.resolve()),method='offline fixed-trajectory reward replay, not a trained v4 policy evaluation',
                discount=.99,strict_outcomes_reproduced=True,stats=stats,trials=trials)
    (output/'reward_audit.json').write_text(json.dumps(result,indent=2))
    lines=['# 奖励离线回放审计','',
        '同一份已保存物理轨迹分别计算v3/v4奖励；不是重新仿真，也不是v4策略效果。严格通过标签已复现。','',
        '|已记录控制|方向数|原通过数|v3存活均奖励为负|v4回放存活均奖励为负|v3最低折扣回报|v4回放最低折扣回报|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode,s in stats.items():lines.append(f"|{mode}|{s['trials']}|{s['recorded_passed']}|{s['v3_negative_live_trials']}|{s['v4_negative_live_trials']}|{s['v3_discounted_min']:.4f}|{s['v4_discounted_min']:.4f}|")
    lines+=['','v4把成本由减分改为正奖励折扣：quality/(1+costs)，保持非负存活奖励；失败一次-5、成功+5不变。',
        '这消除了为规避累计负存活奖励而提前失败的特定激励，不保证PPO收敛或策略改善。不得比较v3/v4奖励数值宣称控制能力提升。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.results,a.output)
