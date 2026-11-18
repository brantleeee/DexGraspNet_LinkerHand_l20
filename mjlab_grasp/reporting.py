"""Summarize completed candidate trials without hiding invalid refinements."""

import json
from pathlib import Path

import numpy as np


def write_report(output, refinement_paths=None):
    output = Path(output)
    results = json.loads((output / "results.json").read_text())
    paths = output.glob("candidate_*/refinement.json") if refinement_paths is None else refinement_paths
    reports = [(p.parent.name, json.loads(p.read_text())) for p in sorted(paths)]
    pairs = []
    lookup = {(r["candidate"], r["variant"], r["case"], r["backend"]): r for r in results}
    for row in results:
        if row["backend"] != "mjlab":
            continue
        cpu = lookup.get((row["candidate"], row["variant"], row["case"], "cpu"))
        if cpu is None:
            continue
        with np.load(output / row["trajectory"]) as gpu_state, np.load(output / cpu["trajectory"]) as cpu_state:
            error = {key: float(np.max(np.abs(gpu_state[key] - cpu_state[key]), initial=0.))
                     for key in ("initial_qpos", "initial_qvel", "initial_ctrl", "initial_mocap_pos", "initial_mocap_quat")}
        pairs.append({"candidate": row["candidate"], "variant": row["variant"], "case": row["case"],
            "initial_state_max_abs_difference": error,
            "verdict_agrees": row["passed"] == cpu["passed"],
            "dynamics_verdict_agrees": row["dynamics_pass"] == cpu["dynamics_pass"],
            "max_slip_difference_mm": abs(row["max_relative_slip_mm"] - cpu["max_relative_slip_mm"]),
            "max_rotation_difference_deg": abs(row["max_rotation_deg"] - cpu["max_rotation_deg"])})
    (output / "backend_comparison.json").write_text(json.dumps(pairs, indent=2, allow_nan=False) + "\n")
    rejected = sum(r["termination"] == "contact_mapping_rejected" for _, r in reports)
    trust = sum(r["termination"] == "trust_region_exceeded" for _, r in reports)
    motion = sorted({r.get("wrist_motion", "mocap") for r in results})
    motion_note = ("手腕使用规定的 freejoint 位置、速度和加速度，理想手臂根部 armature=1e6；不验证真实机械臂能力。"
                   if motion == ["kinematic"] else "包含历史 mocap 位姿更新协议；该协议缺少切向运动速度，lift 结果只能用于复现，不能作为可靠的摩擦携带验收。")
    lines = ["# mjlab 抓取候选细化与动力学筛选", "",
        f"候选数：{len({r['candidate'] for r in results})}；试验数：{len(results)}；严格通过数：{sum(r['passed'] for r in results)}。",
        "", f"接触映射拒绝 {rejected} 个；局部范围超限退出 {trust} 个；细化审计通过 {sum(r['passed'] for _, r in reports)} 个。",
        "", "A/B 共用固定 PD；细化固定物体，验证重新创建自由物体场景。",
        ("CPU 对照验证同一份细化姿态。" if pairs else "本次未运行 CPU 对照。") + "默认保留历史手部自碰撞 mask，未通过的细化姿态仅用于诊断。",
        motion_note,
        "", "| 候选 | 构型 | 后端 | case | 滑移 mm | 旋转 ° | 穿透 mm | 通过 |",
        "|---|---|---|---|---:|---:|---:|---|"]
    for row in results:
        lines.append(f"| {row['candidate']} | {row['variant']} | {row['backend']} | {row['case']} | "
            f"{row['max_relative_slip_mm']:.2f} | {row['max_rotation_deg']:.2f} | {row['peak_object_contact_depth_mm']:.2f} | {row['passed']} |")
    lines += ["", "## 结果解释", "",
        "接触距离改善不等于动态抓稳。几何审计、关节合法性、全过程穿透、逐指承载与末尾连续稳定保持分别检查。",
        "大滑移值可能包含无地面场景中的持续自由落体，不代表候选一直保持接触。",
        "CPU/GPU 初态差异与逐候选判定对照见 backend_comparison.json；失稳或掉落后的轨迹不要求数值重合。",
        "", "本次执行的 case：" + ", ".join(sorted({r["case"] for r in results})) + "。未执行的方向不构成通过证明。",
        "", "本入口尚未实现接触力 QP、自适应夹持或额外 squeeze。当前接触目标由既有标注候选点重建，并非生成器保存的原始接触模板。",
        "动力学中若出现明显 mimic 偏差或限位偏差，应先核查机械约束与控制器，不能只靠增加夹持力解释失败。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
