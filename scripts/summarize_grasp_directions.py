"""Reaggregate audited per-direction engine comparison, retaining partial successes."""
import argparse,csv,json,html
from pathlib import Path
from collections import Counter


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    x=json.loads(a.comparison.read_text());assert x['source_identity_checked']
    rows=[dict(r,replay=str((a.comparison.parent/r['replay']).resolve())) for r in x['rows']];directions=x['directions'];cases=x['direction_order']
    assert len({(r['object'],r['candidate'],r['direction']) for r in directions})==len(rows)*6==len(directions)
    objects=list(dict.fromkeys(r['object'] for r in rows));stats=[];bycase=[];hist=[]
    def stat(group,label):
        result=dict(group=label,trials=len(group))
        for tag,key in [('gym_raw','gym_raw_contact'),('gym_adjusted','gym_adjusted_contact'),('sim','sim_pass')]:
            result[tag]=sum(bool(r[key]) for r in group)
            if tag!='sim':result[tag+'_with_static_gate']=sum(bool(r[key] and r['E_pen_pass']) for r in group)
        return result
    for obj in objects+['ALL']:
        group=[r for r in directions if obj=='ALL' or r['object']==obj];stats.append(stat(group,obj))
        for c in cases:bycase.append(stat([r for r in group if r['direction']==c],obj+'/'+c))
        candidates=[r for r in rows if obj=='ALL' or r['object']==obj]
        for n in range(7):hist.append(dict(object=obj,passed_directions=n,
            gym_raw=sum(r['gym_raw_contact_count']==n for r in candidates),
            gym_adjusted=sum(r['gym_adjusted_contact_count']==n for r in candidates),
            sim=sum(r['sim_pass_count']==n for r in candidates)))
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'object_statistics.csv',stats);write_csv(out/'direction_statistics.csv',bycase)
    write_csv(out/'success_count_distribution.csv',hist);write_csv(out/'all_480_directions.csv',directions)
    write_csv(out/'all_80_candidates.csv',rows)
    concordance={}
    for tag,key in [('raw','gym_raw_contact'),('adjusted','gym_adjusted_contact')]:
        concordance[tag]=dict(both=sum(r[key] and r['sim_pass'] for r in directions),
            gym_only=sum(r[key] and not r['sim_pass'] for r in directions),
            sim_only=sum(not r[key] and r['sim_pass'] for r in directions),
            neither=sum(not r[key] and not r['sim_pass'] for r in directions))
    summary=dict(source=str(a.comparison.resolve()),statistics=stats,by_direction=bycase,distribution=hist,concordance=concordance,full_candidate_stats=x['stats'])
    (out/'statistics.json').write_text(json.dumps(summary,indent=2))
    lines=['# Gym与Isaac Sim：保留部分成功的逐方向统计','',
        '同一批L20的80个原始候选（每物体16个），每候选六方向，共480个方向；源位姿、scale、URDF与网格已重新审计。不是RRL五平移结果。',
        'Gym为官方协议的L20适配，raw表示不调整、adjusted表示静态接触调整；两者列值为数值有效且末帧接触。Sim为既有acquire/10秒严格验收。不同协议，不能当同条件引擎排名。',
        '', '|物体|方向试验数|Gym未调整接触|Gym调整接触|Sim严格通过|','|---|---:|---:|---:|---:|']
    for r in stats:lines.append(f'|{r["group"]}|{r["trials"]}|{r["gym_raw"]}/{r["trials"]}|{r["gym_adjusted"]}/{r["trials"]}|{r["sim"]}/{r["trials"]}|')
    lines+=['','## 静态穿透门限影响','', '|物体|Gym未调整接触且E_pen通过|Gym调整接触且E_pen通过|','|---|---:|---:|']
    for r in stats:lines.append(f'|{r["group"]}|{r["gym_raw_with_static_gate"]}/{r["trials"]}|{r["gym_adjusted_with_static_gate"]}/{r["trials"]}|')
    lines+=['','这里将候选静态E_pen门限附加到各方向，作为诊断口径；官方最终候选成功仍要求六方向全通过且E_pen合格，不能称这些为官方逐方向最终标签。','', '## 合并五物体后的各方向','', '|方向|试验数|Gym未调整接触|Gym调整接触|Sim严格通过|','|---|---:|---:|---:|---:|']
    for r in bycase:
        if r['group'].startswith('ALL/'):lines.append(f'|{r["group"].split("/")[1]}|{r["trials"]}|{r["gym_raw"]}|{r["gym_adjusted"]}|{r["sim"]}|')
    lines+=['','## 每候选成功0–6方向的分布（80候选）','', '|成功方向数|Gym未调整接触|Gym调整接触|Sim严格通过|','|---|---:|---:|---:|']
    for r in hist:
        if r['object']=='ALL':lines.append(f'|{r["passed_directions"]}|{r["gym_raw"]}|{r["gym_adjusted"]}|{r["sim"]}|')
    lines+=['','## 每个候选逐项对照','', '顺序：gx+ gx− gy+ gy− gz+ gz−；1为通过相应判据，0为不通过。Gym静态门限单列。',
        '', '|物体/候选|Gym未调整方向|数量|Gym调整方向|数量|Sim方向|数量|E_pen通过|','|---|---|---:|---|---:|---|---:|---|']
    for r in rows:lines.append(f'|{r["object"]}/{r["candidate"]}|{r["gym_raw_directions"]}|{r["gym_raw_contact_count"]}/6|{r["gym_adjusted_directions"]}|{r["gym_adjusted_contact_count"]}/6|{r["sim_directions"]}|{r["sim_pass_count"]}/6|{r["E_pen_pass"]}|')
    lines+=['','同一候选的六方向相关，480个方向不是480个独立抓取。历史五物体结果与当前RL克隆环境也不能混用；尤其mug0的Gym结果没有组3平移，不能将组3 RRL曲线当作同初态Gym对照。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def table(headers,rr):return '<table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rr)+'</table>'
    page='<!doctype html><meta charset="utf-8"><title>逐方向Gym/Sim对照</title><style>body{font:16px/1.6 system-ui;margin:30px;color:#203040}table{border-collapse:collapse;margin:20px 0}th,td{padding:8px;border:1px solid #ddd}th{background:#edf3f7}</style><h1>80候选 × 六方向：保留部分成功的比较</h1><p>Gym数值有效且末帧接触；Sim为10秒严格验收。协议不同，不能据数量排名引擎。源身份已审计；本页不含RRL新平移。</p><p><a href="REPORT.md">完整Markdown报告</a> · <a href="all_480_directions.csv">480方向CSV</a> · <a href="all_80_candidates.csv">80候选CSV</a> · <a href="statistics.json">统计JSON</a></p>'
    page+=table(['物体','试验数','Gym未调整接触','Gym调整接触','Sim严格通过'],[[r[k] for k in ['group','trials','gym_raw','gym_adjusted','sim']] for r in stats])
    page+='<h2>0–6方向分布</h2>'+table(['成功方向数','Gym未调整','Gym调整','Sim'],[[r[k] for k in ['passed_directions','gym_raw','gym_adjusted','sim']] for r in hist if r['object']=='ALL'])
    page+='<h2>逐候选（方向顺序gx+ gx− gy+ gy− gz+ gz−）</h2>'+table(['物体','候选','Gym未调整','个数','Gym调整','个数','Sim','个数','E_pen通过'],[[r[k] for k in ['object','candidate','gym_raw_directions','gym_raw_contact_count','gym_adjusted_directions','gym_adjusted_contact_count','sim_directions','sim_pass_count','E_pen_pass']] for r in rows])
    (out/'index.html').write_text(page);print(json.dumps(stats,ensure_ascii=False,indent=2));print(out/'index.html')

if __name__=='__main__':main()
