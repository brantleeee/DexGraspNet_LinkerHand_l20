"""Compare saved zero/policy contact timing without simulation."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.contact_timing import summarize_contact_timing
from isaacsim_grasp.isolated import compare_traces


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='Paired group directory containing zero_a and policy')
    p.add_argument('--case',default='gy+')
    p.add_argument('--force-threshold',type=float,default=.001)
    p.add_argument('--sustain-seconds',type=float,default=.05)
    a=p.parse_args();summaries={}
    traces={}
    for control in ['zero_a','policy']:
        directory=a.run/control
        result=json.loads((directory/'result.json').read_text())
        if not result['complete']:raise ValueError('Incomplete trajectory')
        i=result['info']['cases'].index(a.case)
        with np.load(directory/'trajectory.npz') as z:trace={k:z[k] for k in z.files}
        with np.load(directory/'contact_points.npz') as z:
            links=[str(n) for n in z['link_names']];fingers=[str(n) for n in z['finger_names']]
        traces[control]=trace
        s=summarize_contact_timing(trace,links,fingers,i,a.force_threshold,a.sustain_seconds)
        s.update(passed=result['info']['passed'][i],first_failure=result['first_failure'][i])
        # A positive reading elsewhere is a limited sensor sanity check only.
        other=[j for j in range(len(result['info']['cases'])) if j!=i]
        s['other_directions_max_recorded_finger_force_n']=float(trace['finger_normal_force'][:,other,:5].max())
        summaries[control]=s
    prefix=compare_traces(traces['zero_a'],traces['policy'])
    output=dict(case=a.case,source=str(a.run.resolve()),prefix_comparison=prefix,controls=summaries,
        causal_comparison_allowed=prefix['passed'])
    (a.run/'CONTACT_TIMING.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    lines=[f'# {a.case} 接触建立时序诊断','',f'干预前前缀一致：{prefix["passed"]}。诊断阈值仅用于解释记录，不修改抓取验收。',
        '', '候选接触记录不等于有效承载；base_link为掌根。每指连续法向力超过阈值至少50ms（默认）仅表示持续受力证据，不证明夹持闭合。']
    for name,s in summaries.items():
        lines += ['',f'## {name}', '',f'分类：`{s["classification"]}`；严格通过：{s["passed"]}；首次失败：{s["first_failure"]}。',
            '', '|实际时间s|有接触记录的连杆（记录数/法向力N）|各指法向力N|自2.5s位移mm|', '|---|---|---|---:|']
        for x in s['snapshots']:
            records='；'.join(f'{n}: {d["records"]}/{d["normal_force_n"]:.5g}' for n,d in x['links'].items()) or '无'
            forces='；'.join(f'{n}: {d["normal_force_n"]:.5g}' for n,d in x['fingers'].items())
            lines.append(f'|{x["actual_seconds"]:.6f}|{records}|{forces}|{x["displacement_since_action_start_mm"]:.4f}|')
        lines += ['',f'预加载各指峰值法向力N：{s["preload_max_force_n"]}',f'预加载各指最长连续受力秒数：{s["preload_longest_loaded_duration_seconds"]}',
            f'预加载首次/末次受力时刻：{s["preload_load_intervals"]}',
            f'预加载峰值推移：{s["preload_peak_displacement_mm"]:.4f}mm；关节响应：{s["joint_response"]}',
            f'其他方向记录最大手指法向力：{s["other_directions_max_recorded_finger_force_n"]:.5g}N（仅有限传感记录检查）。', '', '2.5秒起接触/受力转换顺序：']
        lines += [f'- {e["time_seconds"]:.6f}s {e["link"]} {e["signal"]} {e["event"]}' for e in s['contact_transitions']] or ['- 无转换记录。']
    lines += ['', '判读：无持续手指受力优先检查接触建立；曾受力但开重力前消失提示预加载接触丢失；开重力时有力随后消失提示加载后脱离。均为记录层面的分类，不断言唯一物理根因。']
    path=a.run/'CONTACT_TIMING.md';path.write_text('\n'.join(lines)+'\n');print(path)
    for name,s in summaries.items():print(name,s['classification'],'max_force',s['preload_max_force_n'],'displacement_mm',s['preload_peak_displacement_mm'],'joint_response',s['joint_response'])


if __name__=='__main__':main()
