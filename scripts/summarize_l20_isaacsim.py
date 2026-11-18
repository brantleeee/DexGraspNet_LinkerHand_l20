"""Audit local object batches and export their reports and offline mesh replays."""
import argparse
import html
import json
from pathlib import Path

from replay_l20_isaacsim_html import export
from l20_object_catalog import OBJECTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--skip-replays',action='store_true')
    p.add_argument('--objects',nargs='+',choices=OBJECTS,help='By default discover all supported object folders in --results')
    a=p.parse_args();root=a.results
    rows=[];expected=0;protocol=None
    objects=a.objects or [name for name in OBJECTS if (root/name/'inputs/manifest.json').exists()]
    if not objects:p.error('No prepared supported object folders found')
    if len(set(objects))!=len(objects):p.error('Duplicate object names')
    scales=set()
    for obj in objects:
        folder=root/obj
        if not (folder/'inputs/manifest.json').exists():continue
        manifest=json.loads((folder/'inputs/manifest.json').read_text())
        if manifest['object']!=obj:raise ValueError(f'Object manifest mismatch: {folder}')
        scales.add(float(manifest['scale']))
        expected+=len(manifest['candidates'])
        batch=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else []
        indices=[item['candidate'] for item in batch]
        if len(set(indices))!=len(indices) or not set(indices)<={c['index'] for c in manifest['candidates']}:
            raise ValueError(f'Duplicate or unknown candidate indices: {folder}')
        for item in batch:
            result=Path(item['output']);row=dict(object=obj,candidate=item['candidate'],completed=item['completed'],output=str(result.resolve()))
            if row['completed']:
                summary=json.loads((result/'summary.json').read_text())
                row['case_isolation']=summary.get('case_isolation','shared_process')
                parameters=summary['parameters']
                current={k:parameters[k] for k in ['steps','dt','sdf_resolution','friction','kp','kd','max_slip','max_angle_deg','max_penetration']}
                current['protocol']=parameters.get('protocol','direct')
                if current['protocol']=='acquire':
                    current.update({k:parameters[k] for k in ['close_seconds','settle_seconds','open_fraction']})
                if protocol is None:protocol=current
                if protocol!=current:raise ValueError(f'Protocol mismatch: {result}')
                stats=json.loads((result/'runtime_collision_statistics.json').read_text())
                audit=json.loads((result/'collision_audit.json').read_text())
                if not stats['available'] or stats['statistics']['nb_convex_shapes']!=0 or stats['statistics']['nb_trimesh_shapes']!=23 or not audit['all_sdf']:
                    raise ValueError(f'Collision audit failed: {result}')
                if summary['manifest']['source_sha256']!=manifest['source_sha256'] or summary['manifest']['meshes']!=manifest['meshes']:
                    raise ValueError(f'Input mismatch: {result}')
                trials=summary['results']
                if len(trials)!=6 or len({t['case'] for t in trials})!=6 or any(not t['numerical_valid'] for t in trials):
                    raise ValueError(f'Incomplete/numerically invalid trials: {result}')
                gains=json.loads((result/'runtime_drives.json').read_text())
                if any(abs(v-5)>1e-5 for v in gains['kp'][0] if v!=0) or any(abs(v-.05)>1e-5 for v in gains['kd'][0] if v!=0):
                    raise ValueError(f'Drive audit failed: {result}')
                row.update(passed_directions=sum(t['passed'] for t in trials),all_directions_pass=all(t['passed'] for t in trials),
                           max_slip_mm=max(t['max_slip_m'] for t in trials)*1000,
                           max_angle_deg=max(t['max_angle_deg'] for t in trials),
                           max_penetration_mm=max(t['max_penetration_m'] for t in trials)*1000,
                           reused=item.get('reused',False))
                replay=root/'visualizations'/f'{obj}_{item["candidate"]:03d}.html'
                newest=max((result/'summary.json').stat().st_mtime,(folder/'inputs/meshes.npz').stat().st_mtime,
                           Path(export.__code__.co_filename).stat().st_mtime)
                if not a.skip_replays and (not replay.exists() or replay.stat().st_mtime<newest):
                    export(result,folder/'inputs',replay)
                row['replay']=str(replay.relative_to(root))
            rows.append(row)
    if (root/'plan.json').exists():expected=json.loads((root/'plan.json').read_text())['expected_candidates']
    completed=sum(r['completed'] for r in rows)
    # Independently launched per-object batches do not have an outer status.json.
    # Completion follows the audited result coverage, not the optional runner status.
    status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else None
    finished=completed==len(rows)==expected and expected>0
    payload=dict(complete=finished,expected_candidates=expected,completed_candidates=completed,
                 directions=completed*6,runner_status=status,rows=rows)
    (root/'summary_all.json').write_text(json.dumps(payload,indent=2))
    intro=('全部完成' if finished else '运行中/尚未完成')+f'：{completed}/{expected} 个候选，{completed*6} 个方向试验。'
    scale_label=', '.join(f'{s:g}' for s in sorted(scales))
    details=(f'scale {scale_label}；每方向 {protocol["steps"]} 步 × {protocol["dt"]:.8f} 秒 = {protocol["steps"]*protocol["dt"]:.3f} 秒；SDF 分辨率 {protocol["sdf_resolution"]}；固定 PD {protocol["kp"]} / {protocol["kd"]}；摩擦系数 {protocol["friction"]}。' if protocol else '尚无完成结果。')
    if protocol and protocol['protocol']=='acquire':
        details+=f'协议 acquire：先零重力平滑闭合 {protocol["close_seconds"]} 秒、保持 {protocol["settle_seconds"]} 秒，再开启重力；不重置物体位姿或速度。'
    elif protocol:details+='协议 direct：历史直接从闭合位姿施加重力，不包含抓取获取。'
    isolated=[f"{r['object']}:{r['candidate']}" for r in rows if r.get('case_isolation')=='fresh_process']
    if isolated:details+=' 以下候选六方向分别使用独立进程（其他候选沿用原场景切换）：'+', '.join(isolated)+'。'
    kinds={json.loads((root/obj/'inputs/manifest.json').read_text()).get('object_mesh_kind') for obj in objects}
    limits=('手部使用完整 STL 的 SDF；物体仍为恢复的旧 CoACD 网格整体建立 SDF，无法恢复原网格细节。' if kinds=={'recovered_coacd_mesh_not_original'} else '手部使用完整STL的SDF；物体几何来源见各inputs/manifest.json。电钻为补内壁/实体并集的非凸源网格，不是CoACD分解。')+'每个完成结果已审计：23 个 triangle-mesh shape，0 个 convex shape。'
    conditions='通过要求：六个方向均满足最大位移≤5 mm、转角≤15°、接触点穿透≤1 mm、末尾连续1秒有接触且数值有效。手腕固定，物体自由，无地面或外部支撑；手部自碰撞及关节速度上限未启用。'
    if protocol and protocol['protocol']=='acquire':conditions+='acquire 还要求开启重力前已有接触；位移和转角相对开启重力前的实际位姿，穿透阈值检查包括闭合在内的全部采样步。'
    md=['# L20 Isaac Sim SDF 抓取验证','',intro,'',details,'',limits,'',conditions,'',
        '|物体|完成候选|六方向全部通过|通过候选下标|','|---|---:|---:|---|']
    overview=[]
    for obj in objects:
        group=[r for r in rows if r['object']==obj and r['completed']]
        valid=[r['candidate'] for r in group if r['all_directions_pass']]
        md.append(f'|{obj}|{len(group)}|{len(valid)}|{valid}|')
        overview.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                        [obj,len(group),len(valid),f'{len(valid)/len(group)*100:.1f}%' if group else '—',valid])+'</tr>')
    md+=['','[打开离线可视化目录](index.html)。回放支持旋转、缩放、分开显示手/物体、切换重力方向和拖动时间轴。',
         '显示 SDF 输入三角网格，不是 SDF 等值面；回放保存的仿真状态，不重新计算物理。','',
         '|物体|候选|通过方向|最大位移 mm|最大转角 °|最大穿透 mm|可视化|','|---|---:|---:|---:|---:|---:|---|']
    htmlrows=[]
    for r in rows:
        if not r['completed']:
            md.append(f'|{r["object"]}|{r["candidate"]}|执行失败|||||')
            htmlrows.append('<tr><td>'+html.escape(r['object'])+'</td><td>'+str(r['candidate'])
                            +'</td><td colspan="5">执行失败／未完成，不计为抓取通过；请检查候选日志。</td></tr>')
            continue
        md.append(f'|{r["object"]}|{r["candidate"]}|{r["passed_directions"]}/6|{r["max_slip_mm"]:.2f}|{r["max_angle_deg"]:.2f}|{r["max_penetration_mm"]:.2f}|[回放]({r["replay"]})|')
        values=[r['object'],str(r['candidate']),f'{r["passed_directions"]}/6',f'{r["max_slip_mm"]:.2f}',f'{r["max_angle_deg"]:.2f}',f'{r["max_penetration_mm"]:.2f}']
        htmlrows.append('<tr class="'+('pass' if r['all_directions_pass'] else '')+'">'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+f'<td><a href="{r["replay"]}">打开回放</a></td></tr>')
    (root/'REPORT.md').write_text('\n'.join(md)+'\n')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>L20 Isaac Sim SDF 验证</title><style>body{font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#182638}p{line-height:1.6}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.pass{background:#e6f5eb}a{color:#125eaf}</style><h1>L20 Isaac Sim SDF 验证</h1>'
        +''.join('<p>'+html.escape(s)+'</p>' for s in [intro,details,limits,conditions,'绿色行表示六方向全部通过。回放显示输入碰撞网格，不是 SDF 等值面。'])
        +'<h2>物体统计</h2><table><thead><tr>'+''.join('<th>'+s+'</th>' for s in ['物体','完成候选','六方向全部通过','通过率','通过候选下标'])+'</tr></thead><tbody>'+''.join(overview)+'</tbody></table><h2>逐候选回放</h2>'
        +'<table><thead><tr>'+''.join('<th>'+s+'</th>' for s in ['物体','候选','通过方向','最大位移 mm','最大转角 °','最大穿透 mm','可视化'])+'</tr></thead><tbody>'+''.join(htmlrows)+'</tbody></table>')
    print(intro)


if __name__=='__main__':main()
