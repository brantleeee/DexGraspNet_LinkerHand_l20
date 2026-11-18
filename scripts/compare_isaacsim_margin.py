"""Compare v2/v3 policies on identical saved perturbations and physical settings."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def compare(current,previous):
    current=Path(current);previous=Path(previous)
    a=json.loads((current/'evaluation.json').read_text());b=json.loads((previous/'evaluation.json').read_text())
    if not a['complete'] or not b['complete'] or a['reference']!=b['reference']:raise ValueError('Incomplete or different reference')
    for name in ['offsets_m','direction_order','coordinate_frame']:
        x=json.loads((current/'perturbations.json').read_text())[name];y=json.loads((previous/'perturbations.json').read_text())[name]
        if x!=y:raise ValueError('Perturbation samples differ')
    ca=json.loads((current/'config.json').read_text());cb=json.loads((previous/'config.json').read_text())
    for k in ['num_envs','position_noise','residual_scale','residual_rate','kp','kd','physics_dt','decimation','close_seconds','settle_seconds','gravity_seconds','open_fraction','collision','mimic']:
        if ca[k]!=cb[k]:raise ValueError(f'Physical/control settings differ: {k}')
    ra=json.loads((current/'reference_snapshot.json').read_text());rb=json.loads((previous/'reference_snapshot.json').read_text())
    for k in ['scene_sha256','source_sha256','targets']:
        if ra[k]!=rb[k]:raise ValueError(f'Reference identity differs: {k}')
    metrics={};pairs=[]
    def keyed(rows):return {(r['episode'],r['env']):r for r in rows}
    for mode in ['zero_residual','policy']:
        ar=keyed(a['runs'][mode]);br=keyed(b['runs'][mode])
        if ar.keys()!=br.keys():raise ValueError('Coverage differs')
        metrics[mode]=dict(previous_pass=sum(r['passed'] for r in br.values()),current_pass=sum(r['passed'] for r in ar.values()))
        for key in ['max_slip_m','max_angle_deg','max_penetration_m','max_mimic_error_rad']:
            old=np.array([br[k][key] for k in sorted(ar)]);new=np.array([ar[k][key] for k in sorted(ar)])
            metrics[mode][key]=dict(previous_median=float(np.median(old)),current_median=float(np.median(new)),
                previous_max=float(old.max()),current_max=float(new.max()),paired_mean_delta=float((new-old).mean()),
                better=int((new<old-1e-8).sum()),worse=int((new>old+1e-8).sum()))
        for k in sorted(ar):
            if ar[k]['case']!=br[k]['case'] or ar[k]['offset_m']!=br[k]['offset_m']:raise ValueError('Trial identity differs')
            if mode=='policy':pairs.append(dict(episode=k[0],env=k[1],case=ar[k]['case'],previous_pass=br[k]['passed'],current_pass=ar[k]['passed']))
    result=dict(previous=str(previous.resolve()),current=str(current.resolve()),same_perturbations=True,physical_settings_unchanged=True,metrics=metrics,pairs=pairs)
    def intervention(c):
        gravity=c['close_seconds']+c['settle_seconds']
        return dict(version=c['version'],action_start_seconds=gravity-c.get('preload_seconds',0.),
                    gravity_start_seconds=gravity)
    result['intervention_timing']=dict(previous=intervention(cb),current=intervention(ca))
    result['evaluation_intervention']=dict(previous=b.get('evaluation_intervention'),current=a.get('evaluation_intervention'))
    (current/'margin_comparison.json').write_text(json.dumps(result,indent=2))
    lines=['# 稳定裕量版本对比','',f'旧结果：{previous}；当前：{current}。','',
        '|控制|指标|旧中位数|当前中位数|旧最大值|当前最大值|逐项改善/恶化|','|---|---|---:|---:|---:|---:|---|']
    for mode,values in metrics.items():
        for key,label,scale in [('max_slip_m','滑移 mm',1000),('max_angle_deg','转角 °',1),('max_penetration_m','穿透 mm',1000),('max_mimic_error_rad','mimic rad',1)]:
            d=values[key];lines.append(f"|{mode}|{label}|{d['previous_median']*scale:.4g}|{d['current_median']*scale:.4g}|{d['previous_max']*scale:.4g}|{d['current_max']*scale:.4g}|{d['better']}/{d['worse']}|")
    lines+=['','先检查通过数不降低，再看连续误差；样本相同不消除跨进程的全部数值差异，应同时检查两次零残差结果是否一致。']
    if ca.get('preload_seconds',0.)!=cb.get('preload_seconds',0.):
        lines+=['',f"动作干预时机已改变：{result['intervention_timing']}。物理与验收设置相同不表示策略干预阶段相同。"]
    if a.get('evaluation_intervention')!=b.get('evaluation_intervention'):
        lines+=['',f"评估控制干预不同：{result['evaluation_intervention']}。需将其解释为固定策略的控制消融。"]
    (current/'MARGIN_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();compare(a.results,a.previous)
