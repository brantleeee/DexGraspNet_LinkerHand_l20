"""Export one offline large figure with paired positions, velocities and joint efforts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.paired_evaluation import CASES


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,float):return round(value,7) if np.isfinite(value) else None
    if isinstance(value,np.generic):return clean(value.item())
    return value


def trial_payload(z,index):
    times=z['time'];gravity=times>=float(z['gravity_start_seconds'])
    reference=np.where(gravity[:,None],z['gravity_reference_position'][index],z['initial_object_position'][index])
    refquat=np.where(gravity[:,None],z['gravity_reference_quaternion'][index],z['object_quaternion'][0,index])
    angle=np.degrees(2*np.arccos(np.clip(np.abs((z['object_quaternion'][:,index]*refquat).sum(-1)),0,1)))
    measured=z['joint_measured_effort'][:,index]
    result=dict(time=times,position=z['object_position'][:,index]*1000,linear=z['object_linear_velocity'][:,index]*1000,
        orientation_error=angle,angular=z['object_angular_velocity'][:,index],
        joint_position=z['joint_position'][:,index],joint_velocity=z['joint_velocity'][:,index],
        tracking_error=z['joint_target'][:,index]-z['joint_position'][:,index],measured=measured,
        target=z['joint_target'][:,index],reference=z['joint_reference'][:,index],pd_estimate=z['joint_pd_estimate'][:,index],
        force=z['contact_force'][:,index],penetration=z['penetration'][:,index]*1000,
        position_error=(z['object_position'][:,index]-reference)*1000,
        peaks=dict(joint_speed_rad_s=float(np.max(abs(z['joint_velocity'][:,index]))),
            measured_joint_effort_Nm=float(np.nanmax(abs(measured))),pd_estimate_Nm=float(np.max(abs(z['joint_pd_estimate'][:,index]))),
            linear_speed_mm_s=float(np.max(np.linalg.norm(z['object_linear_velocity'][:,index],axis=-1)))*1000))
    for key in ['raw_action','requested_residual','effective_requested_residual','applied_residual']:
        if key in z:result[key]=z[key][:,index]
    if 'applied_residual' in z:
        a=z['applied_residual'][times>float(z['gravity_start_seconds']),index]
        result['peaks']['applied_residual_rms_rad']=float(np.sqrt(np.mean(a*a))) if len(a) else 0.
        result['peaks']['applied_residual_abs_bias_rad']=float(np.mean(abs(a.mean(0)))) if len(a) else 0.
    return result


def export(results):
    root=Path(results).resolve();summary=json.loads((root/'evaluation.json').read_text())
    files=summary.get('diagnostics',{})
    if not files or set(files)!=set(['zero_residual','policy']):raise ValueError('Full diagnostics absent: rerun the new paired evaluator; old final-only metrics cannot reconstruct speeds/torques')
    datafolder=root/'diagnostics_data';datafolder.mkdir(exist_ok=True);entries=[];metadata=None
    for episode,(zero_file,policy_file) in enumerate(zip(files['zero_residual'],files['policy'])):
        with np.load(root/zero_file) as zero,np.load(root/policy_file) as policy:
            if not np.array_equal(zero['time'],policy['time']) or not np.array_equal(zero['dof_names'],policy['dof_names']):raise ValueError('Paired diagnostic grids differ')
            if not np.array_equal(zero['cases'],policy['cases']):raise ValueError('Direction views differ')
            names=zero['dof_names'].tolist()
            if metadata is None:metadata=dict(names=names,independent=zero['independent_ids'].tolist(),close=float(zero['close_seconds']),gravity=float(zero['gravity_start_seconds']),end=float(zero['time'][-1]),action_start=summary.get('action_start_seconds'))
            n=zero['joint_position'].shape[1]
            for i in range(n):
                group=summary.get('source_groups',list(range(len(files['zero_residual']))))[episode]*(n//6)+i//6;case=CASES[int(zero['cases'][i])];name=f'group_{group:03d}_{case.replace("+","plus").replace("-","minus")}.js'
                payload=clean(dict(zero_residual=trial_payload(zero,i),policy=trial_payload(policy,i)))
                (datafolder/name).write_text('window.l20TrialData='+json.dumps(payload,separators=(',',':'),allow_nan=False)+';')
                entries.append(dict(label=f'扰动样本 {group} · {case}',file='diagnostics_data/'+name,group=group,case=case,
                    offset_mm=(np.asarray(summary['runs']['zero_residual'][episode*n+i]['offset_m'])*1000).tolist()))
    if len(files['zero_residual'])!=len(files['policy']) or not entries:raise ValueError('Incomplete diagnostic pairing')
    titles=['物体位置 XYZ (mm，原始物体坐标)','物体线速度 XYZ (mm/s)',
        '物体相对参考转角 (deg)','物体角速度 XYZ (rad/s)',
        '全部关节实际位置 (rad)','全部关节实际速度 (rad/s)',
        '关节目标跟踪误差 (rad)','测量投影关节力矩 (N·m，含约束反力)',
        '关节目标位置 (rad)','PD 估算驱动力矩 (N·m，非实测电机输出)',
        '手物接触力指标 (N)','采样接触穿透 (mm)',
        '策略原始动作 (无量纲)','请求残差：scale × tanh(action) (rad)',
        '实际目标残差：限速/投影后 (rad)','阶段/失败屏蔽后的请求残差 (rad)']
    fig=make_subplots(rows=8,cols=2,shared_xaxes='all',subplot_titles=titles,vertical_spacing=.035)
    fig.update_layout(height=2400,hovermode='x unified',margin=dict(l=65,r=30,t=70,b=50),showlegend=False,
        title='零残差 / 策略：位置、速度、力矩统一分析',template='plotly_white')
    for row in range(1,9):
        for col in [1,2]:fig.update_xaxes(title_text='时间 (s)',row=row,col=col)
    layout=json.loads(json.dumps(fig.to_plotly_json()['layout'],cls=PlotlyJSONEncoder))
    manifest=json.dumps(clean(dict(entries=entries,meta=metadata)),ensure_ascii=False,separators=(',',':'))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>L20 配对位置速度力矩大图</title>
<style>body{font:15px system-ui;margin:18px;color:#182638}p{line-height:1.7}select{margin:5px 14px 5px 0;padding:6px}#plot{width:100%;min-width:900px}#container{overflow-x:auto}#peaks{padding:10px;background:#eef3f8}#error{color:#a01818}</style>
<h1>L20：零残差 / 策略，位置·速度·力矩统一分析</h1>
<p>同一个大图包含全部分析，共享时间轴，可缩放并悬停查看。实线为零残差、虚线为策略；关节颜色一致。目标位置面板的点线为原始reference。灰蓝/浅绿背景分别表示闭合/零重力保持，随后为重力验证。</p>
<p><b>测量投影关节力矩不是单独的电机输出</b>，可能含约束反力；PD力矩为采样状态下的限幅估算。起始帧尚未推进物理步，测量力矩留空。所有物理步以240 Hz记录，目标速度不是实际速度。新增四个动作面板区分原始动作、请求、屏蔽与实际目标残差；旧数据缺少的动作不会反推或伪造。</p>
<label>扰动/方向 <select id="trial"></select></label><label>关节 <select id="joint"><option value="all">全部21关节</option><option value="active">16主动关节</option><option value="thumb_">拇指</option><option value="index_">食指</option><option value="middle_">中指</option><option value="ring_">无名指</option><option value="pinky_">小指</option></select></label>
<label>控制组 <select id="mode"><option value="both">两组叠加</option><option value="zero_residual">仅零残差</option><option value="policy">仅策略</option></select></label>
<label>物体位置 <select id="position"><option value="position">实际XYZ位置</option><option value="position_error">相对阶段参考的XYZ误差</option></select></label>
<p>位置误差在零重力阶段相对扰动初态，重力阶段相对开启重力前实测位姿；参考切换不代表物体被重置。峰值按全部240Hz样本计算。请保留 diagnostics_data 文件夹。</p>
<p id="peaks"></p><p id="error"></p><div id="container"><div id="plot"></div></div><script>'''+get_plotlyjs()+'''</script><script>
const manifest=MANIFEST,baseLayout=LAYOUT;
const trial=document.getElementById('trial'),joint=document.getElementById('joint'),mode=document.getElementById('mode'),position=document.getElementById('position');
manifest.entries.forEach((v,i)=>trial.add(new Option(v.label,i)));manifest.meta.names.forEach((v,i)=>joint.add(new Option(v,'joint:'+i)));
let payload=null,loadId=0;
const colors=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2','#be185d','#4f46e5','#65a30d','#a16207','#0f766e','#7e22ce','#475569','#e11d48','#0284c7','#15803d','#b45309','#6d28d9','#0e7490','#9f1239','#334155'];
function selected(j){let v=joint.value;return v==='all'||(v==='active'&&manifest.meta.independent.includes(j))||(v.startsWith('joint:')&&j===Number(v.slice(6)))||manifest.meta.names[j].startsWith(v);}
function draw(){if(!payload)return;let traces=[];const shown=mode.value==='both'?['zero_residual','policy']:[mode.value];
function add(data,m,key,label,axis,color,column=null){traces.push({type:'scattergl',mode:'lines',name:(m==='reference'?'原始参考 ':m==='policy'?'策略 ':'零残差 ')+label,x:data.time,y:column===null?data[key]:data[key].map(r=>r[column]),xaxis:axis===1?'x':'x'+axis,yaxis:axis===1?'y':'y'+axis,line:{color:color,width:1.3,dash:m==='reference'?'dot':m==='policy'?'dash':'solid'},hovertemplate:'%{y:.5g}<extra>%{fullData.name}</extra>'});}
for(const m of shown){let d=payload[m];['X','Y','Z'].forEach((n,j)=>{add(d,m,position.value,n+'位置',1,colors[j],j);add(d,m,'linear',n+'线速度',2,colors[j],j);add(d,m,'angular',n+'角速度',4,colors[j],j);});add(d,m,'orientation_error','转角误差',3,colors[0]);
manifest.meta.names.forEach((n,j)=>{if(!selected(j))return;add(d,m,'joint_position',n,5,colors[j],j);add(d,m,'joint_velocity',n,6,colors[j],j);add(d,m,'tracking_error',n,7,colors[j],j);add(d,m,'measured',n,8,colors[j],j);add(d,m,'target',n,9,colors[j],j);if(manifest.meta.independent.includes(j))add(d,m,'pd_estimate',n,10,colors[j],j);});add(d,m,'force','接触力',11,colors[0]);add(d,m,'penetration','穿透',12,colors[0]);}
for(const m of shown){const d=payload[m];manifest.meta.independent.forEach((j,k)=>{if(!selected(j))return;for(const [key,axis] of [['raw_action',13],['requested_residual',14],['applied_residual',15],['effective_requested_residual',16]]){if(d[key])add(d,m,key,manifest.meta.names[j],axis,colors[j],k);}});}
manifest.meta.names.forEach((n,j)=>{if(selected(j))add(payload.zero_residual,'reference','reference',n,9,colors[j],j);});
let layout=JSON.parse(JSON.stringify(baseLayout));layout.shapes=[];
for(const axis of ['x','x2']){for(const band of [[0,manifest.meta.close,'#bfdbfe'],[manifest.meta.close,manifest.meta.gravity,'#bbf7d0']])layout.shapes.push({type:'rect',xref:axis,yref:'paper',x0:band[0],x1:band[1],y0:0,y1:1,fillcolor:band[2],opacity:.18,line:{width:0},layer:'below'});for(const x of [manifest.meta.close,manifest.meta.action_start,manifest.meta.gravity].filter(v=>v!==null&&v!==undefined))layout.shapes.push({type:'line',xref:axis,yref:'paper',x0:x,x1:x,y0:0,y1:1,line:{color:'#64748b',width:1,dash:'dot'}});}
layout.title={text:manifest.entries[Number(trial.value)].label+' · 位置 / 速度 / 力矩'};layout.uirevision='same-grid';
layout.annotations[0].text=position.value==='position'?'物体位置 XYZ (mm，原始物体坐标)':'物体位置误差 XYZ (mm，相对阶段参考)';
Plotly.react('plot',traces,layout,{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:'l20_paired_diagnostics',width:1600,height:2450,scale:1.5}});
let text=['扰动XYZ (mm)：'+manifest.entries[Number(trial.value)].offset_mm.map(v=>v.toFixed(4)).join(', ')];for(const m of shown){let p=payload[m].peaks;text.push((m==='policy'?'策略':'零残差')+'：全关节最大|速度| '+p.joint_speed_rad_s.toFixed(4)+' rad/s；最大|测量投影力矩| '+p.measured_joint_effort_Nm.toFixed(4)+' N·m；最大|PD估算| '+p.pd_estimate_Nm.toFixed(4)+' N·m');}document.getElementById('peaks').textContent=text.join(' ｜ ');}
function load(){let id=++loadId;document.getElementById('error').textContent='加载数据…';let s=document.createElement('script');s.src=manifest.entries[Number(trial.value)].file;s.onload=()=>{if(id===loadId){payload=window.l20TrialData;document.getElementById('error').textContent='';draw();}s.remove();};s.onerror=()=>{document.getElementById('error').textContent='数据加载失败，请保留 diagnostics_data 与 HTML 的相对位置。';s.remove();};document.body.appendChild(s);}
trial.addEventListener('change',load);joint.addEventListener('change',draw);mode.addEventListener('change',draw);position.addEventListener('change',draw);load();
</script></html>'''
    page=page.replace('const manifest=MANIFEST,baseLayout=LAYOUT;',f'const manifest={manifest},baseLayout={json.dumps(layout,cls=PlotlyJSONEncoder)};')
    destination=root/'diagnostics.html';destination.write_text(page)
    print(f'Unified diagnostics: {destination} ({len(entries)} paired trials)')
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    export(p.parse_args().results)


if __name__=='__main__':main()
