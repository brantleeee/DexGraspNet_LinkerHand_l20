"""Build paired six-direction diagnostics and publishable curves from completed runs."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
from isaacsim_grasp.isolated import compare_traces
from plot_isaacsim_paired_diagnostics import export


def angle(q,ref):
    q=np.asarray(q,float);ref=np.asarray(ref,float)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);ref=ref/np.linalg.norm(ref,axis=-1,keepdims=True)
    return np.degrees(2*np.arccos(np.clip(abs(np.sum(q*ref,axis=-1)),0,1)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--zero',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    traces={};results={}
    for key,run in [('zero_residual',a.zero),('policy',a.policy)]:
        with np.load(run/'trajectory.npz') as z:traces[key]={k:z[k] for k in z.files}
        results[key]=json.loads((run/'result.json').read_text())
        assert results[key]['reference']['object']=='mug' and results[key]['reference']['candidate']==0
        assert results[key]['source_group']==3
    prefix=compare_traces(traces['zero_residual'],traces['policy']);assert prefix['passed']
    plan=json.loads((a.zero/'plan.json').read_text());offsets=plan['samples']['offsets_m'][0]
    adapter=dict(schema='plot_adapter_not_new_evaluation',source_groups=[3],action_start_seconds=2.5,
        diagnostics={k:[str((r/'trajectory.npz').resolve())] for k,r in [('zero_residual',a.zero),('policy',a.policy)]},
        runs=dict(zero_residual=[dict(offset_m=v) for v in offsets]),source_runs=dict(zero=str(a.zero.resolve()),policy=str(a.policy.resolve())),prefix_comparison=prefix)
    (out/'evaluation.json').write_text(json.dumps(adapter,indent=2));export(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(14,11),sharex=True);colors={'zero_residual':'#8c4d39','policy':'#007d76'}
    metrics={};csvrows=[]
    for key,z in traces.items():
        t=z['time'];g=float(z['gravity_start_seconds']);mask=t>=g;cols=[]
        for j,c in enumerate(results[key]['info']['cases']):
            slip=np.linalg.norm(z['object_position'][:,j]-z['gravity_reference_position'][j],axis=-1)*1000
            ang=angle(z['object_quaternion'][:,j],z['gravity_reference_quaternion'][j])
            metrics[(key,c)]=(slip,ang)
            ax=axs.flat[j];ax.plot(t[mask],slip[mask],label=key,color=colors[key],linewidth=1.3)
            ax.set_yscale('symlog',linthresh=1.);ax.axhline(5,color='#bb3535',linestyle='--',linewidth=.8)
            ax.set_title(c);ax.set_ylabel('Slip (mm), symlog');ax.grid(alpha=.2);ax.set_xlim(3,13)
            for k in range(len(t)):
                csvrows.append(dict(controller=key,direction=c,time_s=float(t[k]),phase=int(z['phase'][k,j]),
                    slip_from_gravity_reference_mm=float(slip[k]),angle_from_gravity_reference_deg=float(ang[k]),
                    force_metric_n=float(z['contact_force'][k,j]),penetration_mm=float(z['penetration'][k,j]*1000)))
    for ax in axs[-1]:ax.set_xlabel('Simulation time (s)')
    axs.flat[0].legend();fig.suptitle('Mug 0 / original group 3: six-direction slip vs 5 mm limit\nSaved 240 Hz live-validation data; symlog preserves drop tails');fig.tight_layout();fig.savefig(out/'six_direction_slip.png',dpi=170);plt.close(fig)
    for c in ['gy+','gz+']:
        j=results['policy']['info']['cases'].index(c);fig,axs=plt.subplots(3,2,figsize=(14,10),sharex=True)
        for key,z in traces.items():
            t=z['time'];slip,ang=metrics[(key,c)];col=colors[key];ids=np.asarray(z['independent_ids'],int)
            axs[0,0].plot(t,slip,label=key,color=col);axs[0,1].plot(t,ang,color=col)
            axs[1,0].plot(t,z['contact_force'][:,j],color=col);axs[1,1].plot(t,z['penetration'][:,j]*1000,color=col)
            rms=np.sqrt(np.mean(z['applied_residual'][:,j]**2,axis=-1));qd=np.sqrt(np.mean(z['joint_velocity'][:,j][:,ids]**2,axis=-1))
            axs[2,0].plot(t,rms,color=col);axs[2,1].plot(t,qd,color=col)
            failure=results[key]['first_failure'][j]
            if failure:
                for ax in axs.flat:ax.axvline(failure['time_seconds'],color=col,linestyle=':',alpha=.7)
        axs[0,0].set_yscale('symlog',linthresh=1);axs[0,0].axhline(5,color='red',linestyle='--',linewidth=.8)
        axs[0,1].axhline(15,color='red',linestyle='--',linewidth=.8);axs[1,1].axhline(1,color='red',linestyle='--',linewidth=.8)
        titles=['Slip from gravity reference (mm; symlog)','Angle from gravity reference (deg)','Normal contact-force metric (N)','Sampled penetration (mm)','Applied independent residual RMS (rad)','Actual independent joint speed RMS (rad/s)']
        for ax,title in zip(axs.flat,titles):
            ax.set_title(title);ax.grid(alpha=.2);ax.set_xlim(2.4,13)
            for edge in [2.5,3]:ax.axvline(edge,color='#586777',linestyle='--',linewidth=.7)
        for ax in axs[-1]:ax.set_xlabel('Simulation time (s)')
        axs[0,0].legend();fig.suptitle(f'Mug 0 / original group 3 / {c}: zero residual vs frozen RRL\n2.5 s action start; 3 s gravity start; dotted colored line = first strict failure');fig.tight_layout();fig.savefig(out/f'{c.replace("+","plus")}_curves.png',dpi=170);plt.close(fig)
    with (out/'object_curves_240hz.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    (out/'RESULTS.json').write_text(json.dumps(dict(reference='mug original candidate 0, group 3',prefix_comparison=prefix,results=results),indent=2))
    page='<!doctype html><meta charset="utf-8"><title>Mug0组3曲线</title><style>body{font:16px/1.6 system-ui;margin:30px;color:#203040}img{max-width:100%}</style><h1>杯子0 · 原始扰动组3 · 零残差/RRL曲线</h1><p>本次在线验证的实际240Hz记录；零残差4/6、RRL6/6，干预前前缀一致。不是Gym同初态对照。坐标为未旋转物体网格的初始坐标轴；滑移/转角相对3秒实际物体姿态。全程图在重力前的误差仅供参考，不用于重力阈值验收。</p><p><a href="diagnostics.html">交互曲线：六方向、全部关节、位置/速度/力矩/目标/残差</a> · <a href="object_curves_240hz.csv">物体曲线CSV</a> · <a href="RESULTS.json">结果和来源</a></p><p>对称对数轴保留失败后的掉落大值，未截断源数据；视频/图上的最终恢复不消除历史失败。测量关节力矩包含约束反力，PD估算不是实测电机输出。</p>'
    for name in ['six_direction_slip.png','gyplus_curves.png','gzplus_curves.png']:page+='<img src="'+name+'">'
    (out/'index.html').write_text(page);print(out/'index.html')

if __name__=='__main__':main()
