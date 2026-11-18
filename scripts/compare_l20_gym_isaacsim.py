"""Audit and compare each original L20 candidate and gravity direction across saved engines."""
import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path

import numpy as np
from l20_object_catalog import OBJECTS,ROOT

DIRECTIONS=['gx+','gx-','gy+','gy-','gz+','gz-']


def gym_direction_map():
    # Object-local gravity is R.T @ world_gravity. These are upstream test_rotations.
    rotations=[('z',0),('z',math.pi),('z',math.pi/2),('z',-math.pi/2),('x',math.pi/2),('x',-math.pi/2)]
    result={}
    for i,(axis,angle) in enumerate(rotations):
        c,s=math.cos(angle),math.sin(angle)
        r=np.array([[c,-s,0],[s,c,0],[0,0,1]]) if axis=='z' else np.array([[1,0,0],[0,c,-s],[0,s,c]])
        gravity=r.T@np.array([0.,-1.,0.]);j=int(np.argmax(np.abs(gravity)))
        result[i]='g'+'xyz'[j]+('+' if gravity[j]>0 else '-')
    return result


def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gym-results',type=Path,required=True)
    p.add_argument('--sim-results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();simroot=a.sim_results.resolve();gymroot=a.gym_results.resolve()
    if out.exists():p.error('Output must be new')
    gym=json.loads((gymroot/'summary.json').read_text());sim=json.loads((simroot/'summary_all.json').read_text())
    if not gym['completed'] or not sim['complete']:raise ValueError('Both input batches must be complete')
    upstream=ROOT/'external/DexGraspNet_official/grasp_generation/utils/isaac_validator.py'
    if hashlib.sha256(upstream.read_bytes()).hexdigest()!=gym['source_validator_sha256']:
        raise ValueError('Upstream validator hash changed; recheck rotation mapping')
    mapping=gym_direction_map();lookup={};rows=[];details=[]
    for run in gym['runs']:
        for c in run['candidates']:
            key=(run['object'],c['index'],run['variant'])
            if key in lookup:raise ValueError(f'Duplicate Gym candidate: {key}')
            lookup[key]=c
    for item in sim['rows']:
        if not item['completed']:raise ValueError('Incomplete candidate')
        obj=item['object'];i=item['candidate'];result=Path(item['output'])
        saved=json.loads((result/'summary.json').read_text());manifest=saved['manifest']
        old=gym['input_manifest']['objects'][obj]['source_manifest']
        for key in ['object','scale','source_sha256','urdf_sha256','meshes','candidates']:
            if manifest[key]!=old[key]:raise ValueError(f'Source mismatch: {obj} {i} {key}')
        records={t['case']:t for t in saved['results']}
        if set(records)!=set(DIRECTIONS):raise ValueError('Invalid Sim direction coverage')
        row=dict(object=obj,candidate=i,scale=manifest['scale'])
        gc={}
        for variant in ['official','no-force']:
            c=lookup[(obj,i,variant)]
            if abs(c['scale']-manifest['scale'])>1e-7:raise ValueError('Scale mismatch')
            dirs={mapping[d['rotation_index']]:d for d in c['directions']}
            if len(c['directions'])!=6 or set(dirs)!=set(DIRECTIONS):raise ValueError('Invalid Gym direction coverage')
            passed=[bool(dirs[d]['final_contact'] and dirs[d]['numerical_valid']) for d in DIRECTIONS]
            if c['simulated_pass']!=all(passed) or c['valid']!=(all(passed) and c['estimated_pass']):raise ValueError('Gym flag mismatch')
            gc[variant]=(c,dirs)
            prefix='gym_adjusted' if variant=='official' else 'gym_raw'
            row[prefix+'_directions']=''.join('1' if v else '0' for v in passed)
            row[prefix+'_contact_count']=sum(passed)
            row[prefix+'_contact_all']=c['simulated_pass'];row[prefix+'_valid']=c['valid']
        if gc['official'][0]['E_pen']!=gc['no-force'][0]['E_pen']:raise ValueError('Saved E_pen mismatch')
        row['saved_E_pen']=gc['no-force'][0]['E_pen'];row['E_pen_pass']=gc['no-force'][0]['estimated_pass']
        row['sim_directions']=''.join('1' if records[d]['passed'] else '0' for d in DIRECTIONS)
        row['sim_pass_count']=sum(records[d]['passed'] for d in DIRECTIONS)
        row['sim_all']=all(records[d]['passed'] for d in DIRECTIONS)
        if row['sim_all']!=saved['all_directions_pass']:raise ValueError('Sim flag mismatch')
        row['case_isolation']=saved.get('case_isolation','shared_process')
        row['replay']=os.path.relpath(simroot/item['replay'],out)
        if not (simroot/item['replay']).is_file():raise ValueError('Missing Sim replay')
        for d in DIRECTIONS:
            t=records[d];raw=gc['no-force'][1][d];adjusted=gc['official'][1][d]
            details.append(dict(object=obj,candidate=i,direction=d,saved_E_pen=row['saved_E_pen'],E_pen_pass=row['E_pen_pass'],
                gym_raw_contact=bool(raw['final_contact'] and raw['numerical_valid']),gym_raw_contact_points=raw['final_contact_count'],
                gym_adjusted_contact=bool(adjusted['final_contact'] and adjusted['numerical_valid']),gym_adjusted_contact_points=adjusted['final_contact_count'],
                gym_raw_mimic_error_rad=raw['final_max_mimic_error_rad'],gym_adjusted_mimic_error_rad=adjusted['final_max_mimic_error_rad'],
                sim_pass=t['passed'],sim_acquired=t['acquired_before_gravity'],sim_max_slip_mm=t['max_slip_m']*1000,
                sim_max_angle_deg=t['max_angle_deg'],sim_max_penetration_mm=t['max_penetration_m']*1000,
                sim_continuous_last_second_contact=t['continuous_last_second_contact'],sim_numerical_valid=t['numerical_valid']))
        rows.append(row)
    expected={(r['object'],r['candidate'],v) for r in rows for v in ['official','no-force']}
    if set(lookup)!=expected:raise ValueError('Candidate coverage differs across engines')
    objects=[o for o in OBJECTS if any(r['object']==o for r in rows)]
    rows.sort(key=lambda r:(objects.index(r['object']),r['candidate']))
    stats=[]
    for obj in objects+['ALL']:
        group=[r for r in rows if obj=='ALL' or r['object']==obj]
        stat=dict(object=obj,candidates=len(group),gym_raw_contact_all=sum(r['gym_raw_contact_all'] for r in group),
            gym_raw_valid=sum(r['gym_raw_valid'] for r in group),gym_adjusted_contact_all=sum(r['gym_adjusted_contact_all'] for r in group),
            gym_adjusted_valid=sum(r['gym_adjusted_valid'] for r in group),sim_all=sum(r['sim_all'] for r in group))
        for variant in ['raw','adjusted']:
            g='gym_'+variant+'_valid'
            stat[variant+'_both']=sum(r[g] and r['sim_all'] for r in group)
            stat[variant+'_gym_only']=sum(r[g] and not r['sim_all'] for r in group)
            stat[variant+'_sim_only']=sum(not r[g] and r['sim_all'] for r in group)
            stat[variant+'_neither']=sum(not r[g] and not r['sim_all'] for r in group)
        stats.append(stat)
    out.mkdir(parents=True)
    payload=dict(gym_results=str(gymroot),sim_results=str(simroot),direction_order=DIRECTIONS,gym_rotation_index_to_local_gravity=mapping,
        gym_steps=gym['sim_steps'],source_identity_checked=True,stats=stats,rows=rows,directions=details)
    (out/'comparison.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    save_csv(out/'candidates.csv',rows);save_csv(out/'directions.csv',details)
    note='方向按物体原始坐标统一为 gx+、gx−、gy+、gy−、gz+、gz−；✓通过，×未通过。Gym方向格仅为末帧接触且数值有效；E_pen独立列出，Gym最终通过要求六格全✓且E_pen通过。'
    limits='两边不是同一验收条件：Gym 100步≈1.67秒，从闭合姿态开始、凸碰撞、21位置驱动、mimic仅一致目标；Isaac Sim零重力闭合2秒+保持1秒后重力10秒、SDF、16驱动+5硬mimic，并检查位移/转角/穿透/连续接触。Gym调整组还改变了关节目标，所以原始组是位姿更直接的对应，但也不是仅更换引擎的控制实验。'
    md=['# L20 五物体：逐位姿、逐方向 Gym / Isaac Sim 对比','',note,'',limits,'',
        '数据已按物体、原始 NPY 下标、比例、源 NPY/URDF/网格散列及候选清单核对。',
        '相机候选3的 Isaac Sim 结果使用每方向独立进程，其他候选沿用原记录。','',
        '|物体|候选|Gym原始六方向接触|Gym原始最终通过|Gym调整六方向接触|Gym调整最终通过|Sim六方向严格通过|',
        '|---|---:|---:|---:|---:|---:|---:|']
    overview=[]
    for s in stats:
        values=[s[k] for k in ['object','candidates','gym_raw_contact_all','gym_raw_valid','gym_adjusted_contact_all','gym_adjusted_valid','sim_all']]
        md.append('|'+ '|'.join(map(str,values))+'|')
        overview.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>')
    md+=['','## 每个候选','', '|物体|原始下标|Gym原始六方向|Gym原始最终|Gym调整六方向|Gym调整最终|Sim六方向|Sim最终|E_pen通过|回放|',
         '|---|---:|---|---|---|---|---|---|---|---|']
    def yn(v):return '✓' if v else '×'
    def bits(v):return ''.join(yn(x=='1') for x in v)
    def badges(v):return '<span class="bits">'+''.join('<b class="'+('yes' if x=='1' else 'no')+'" title="'+d+'">'+yn(x=='1')+'</b>' for d,x in zip(DIRECTIONS,v))+'</span>'
    detail_by={(r['object'],r['candidate'],r['direction']):r for r in details}
    htmlrows=[]
    for r in rows:
        md.append(f"|{r['object']}|{r['candidate']}|{bits(r['gym_raw_directions'])}|{yn(r['gym_raw_valid'])}|{bits(r['gym_adjusted_directions'])}|{yn(r['gym_adjusted_valid'])}|{bits(r['sim_directions'])}|{yn(r['sim_all'])}|{yn(r['E_pen_pass'])}|[Sim回放]({r['replay']})|")
        tr='<tr data-object="'+r['object']+'" data-different="'+str(r['gym_raw_valid']!=r['sim_all']).lower()+'">'
        tr+='<td>'+r['object']+'</td><td>'+str(r['candidate'])+'</td>'
        for prefix in ['gym_raw','gym_adjusted','sim']:
            final=r['sim_all'] if prefix=='sim' else r[prefix+'_valid']
            tr+='<td>'+badges(r[prefix+'_directions'])+'</td><td>'+yn(final)+'</td>'
        tr+='<td>'+yn(r['E_pen_pass'])+f"<br><small>{r['saved_E_pen']:.6g}</small></td>"
        inner=[]
        for d in DIRECTIONS:
            t=detail_by[(r['object'],r['candidate'],d)]
            values=[d,t['gym_raw_contact_points'],t['gym_adjusted_contact_points'],f"{t['sim_max_slip_mm']:.2f}",f"{t['sim_max_angle_deg']:.2f}",f"{t['sim_max_penetration_mm']:.3f}",yn(t['sim_acquired']),yn(t['sim_continuous_last_second_contact'])]
            inner.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>')
        tr+='<td><a href="'+html.escape(r['replay'],quote=True)+'">Sim回放</a><details><summary>方向指标</summary><table><tr>'+''.join('<th>'+v+'</th>' for v in ['方向','Gym原始接触数','Gym调整接触数','Sim位移mm','Sim转角°','Sim穿透mm','Sim获取','Sim末秒接触'])+'</tr>'+''.join(inner)+'</table></details></td></tr>'
        htmlrows.append(tr)
    md+=['','逐方向完整数据（含 Gym 实际 mimic 偏差）见 directions.csv；完整原始下标对齐结果见 comparison.json。']
    (out/'REPORT.md').write_text('\n'.join(md)+'\n')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>L20 Gym / Isaac Sim 逐位姿对比</title><style>body{font:15px system-ui;color:#182638;margin:30px}p{max-width:1200px;line-height:1.7}table{border-collapse:collapse;margin:18px 0}td,th{padding:9px;border:1px solid #ddd;text-align:left}th{background:#eef2f7}.bits{white-space:nowrap}.bits b{display:inline-block;text-align:center;width:23px;padding:4px 0;margin:1px;border-radius:3px}.yes{background:#dbf2e2;color:#17612e}.no{background:#fae0e0;color:#9b2424}small{color:#637080}details table{font-size:12px}a{color:#155fc0}select,label{margin-right:16px}details{max-width:650px;overflow-x:auto}</style><h1>L20：80个位姿、六方向逐项对比</h1>'
    page+=''.join('<p>'+html.escape(t)+'</p>' for t in [note,limits,'已核对源位姿身份；相机候选3在Sim中采用每方向独立进程。此页面展示已记录结果，不重新仿真。'])
    page+='<p><a href="candidates.csv">80候选CSV</a> · <a href="directions.csv">480方向CSV</a> · <a href="comparison.json">完整JSON</a> · <a href="REPORT.md">报告</a></p><h2>总览</h2><table><tr>'+''.join('<th>'+v+'</th>' for v in ['物体','候选数','Gym原始接触6/6','Gym原始最终','Gym调整接触6/6','Gym调整最终','Sim严格6/6'])+'</tr>'+''.join(overview)+'</table>'
    page+='<h2>逐位姿</h2><p><select id="object"><option value="">全部物体</option>'+''.join('<option>'+o+'</option>' for o in objects)+'</select><label><input id="different" type="checkbox">仅看 Gym原始最终 / Sim最终 不一致</label><span id="count"></span></p><p>每组方向顺序固定：gx+、gx−、gy+、gy−、gz+、gz−。鼠标悬停色块可查看方向。</p><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in ['物体','原始下标','Gym原始接触方向','Gym原始最终','Gym调整接触方向','Gym调整最终','Sim严格方向','Sim最终','E_pen通过','指标/回放'])+'</tr></thead><tbody id="poses">'+''.join(htmlrows)+'</tbody></table>'
    page+='<script>const obj=document.getElementById("object"),diff=document.getElementById("different");function filter(){let n=0;for(const r of document.querySelectorAll("#poses>tr")){const show=(!obj.value||r.dataset.object===obj.value)&&(!diff.checked||r.dataset.different==="true");r.hidden=!show;if(show)n++;}document.getElementById("count").textContent=n+" / '+str(len(rows))+' 个候选";}obj.addEventListener("change",filter);diff.addEventListener("change",filter);filter();</script></html>'
    (out/'index.html').write_text(page)
    print(json.dumps(stats,ensure_ascii=False,indent=2));print(f'Comparison: {out}/index.html')


if __name__=='__main__':main()
