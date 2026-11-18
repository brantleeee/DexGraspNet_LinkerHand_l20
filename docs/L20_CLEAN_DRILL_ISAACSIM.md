# L20 clean电钻抓取：Isaac Sim SDF动力学验证

入口：`scripts/validate_l20_clean_drill.py`。2026-11-15本轮只撰写代码、CPU准备和测试，没有启动批量仿真。

## 输入与默认协议

源批次l20_clean_drill_20261115已完成32候选，全部URDF/mimic可行，8候选通过静态几何筛选；按能量前16中有7个几何通过。程序默认验证前16所有候选，不再筛掉几何未通过项，报告保留原下标/能量排名/几何标记。排名SHA、源NPY SHA、mesh identity、保存比例1、每候选能量/索引均核对。

源model_clean.obj是开放抓取表面。程序从冻结的生成网格制作独立完整非凸实体代理：补匹配孔壁80面和实体并集，无凸包/凸分解/减面，无坐标/轴/比例变换。原面和姿态不变。将代理用于SDF，不宣称原clean网格原封不动成为闭合体。

- 完整手部L20网格＋完整物体SDF256，16独立位置驱动＋5物理mimic约束。
- 固定手根，自由物体，无支撑/焊接；每方向独立原生Isaac Sim进程。
- 物体质量按用户确认600g；惯量为均匀包围盒近似，非实测。
- 默认acquire：零重力闭合2秒、稳定1秒，随后六方向重力各10秒。开重力不重置物体位姿或速度。
- 可选direct：直接加载保存的闭合位姿，立即开重力10秒。两种协议不混合统计。
- 默认Kp5、Kd0.05、独立关节力矩上限1Nm、摩擦3、dt1/240。可显式指定--kp/--kd/--effort-limit做独立对照；增上限不等于接触力自动增大。

## 完整验证命令

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

l20_check="data/experiments/l20_clean_drill_sdf_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/validate_l20_clean_drill.py \
  --ranking data/experiments/l20_clean_drill_20261115/results/scales/scale_1/top_16/ranking.json \
  --top-k 16 --protocol acquire \
  --kp 5 --kd 0.05 --effort-limit 1 \
  --sdf-resolution 256 --output "$l20_check" \
  > "$l20_check.log" 2>&1

# 完成后在同一终端查看，变量仅在该终端有效
cat "$l20_check/REPORT.md"
xdg-open "$l20_check/index.html"
```

如只测原静态位姿，把--protocol acquire改为--protocol direct，并使用新输出目录。

## 准备检查和复用已准备输入

`--prepare-only`仅核对/生成碰撞输入和计划，不启动Isaac Sim。CPU准备完成后，如要使用同一目录运行，加--prepared-run，其他参数保持完全相同；会验证manifest、NPZ、原NPY、排名和碰撞网格散列。

本轮准备检查目录：data/experiments/l20_clean_drill_sdf_preparecheck_20261115，16候选/96预期方向，23个SDF输入网格、22手部连杆、16驱动、5mimic、质量0.6kg。这个目录不包含已执行动力学结果，不能算通过。

程序不自动续跑。有results.json的目录会拒绝prepared-run；需要新批次或另行实现审计续跑，不覆盖历史。Isaac Sim共用进程锁，其他任务占锁时排队，不停止它们。

## 验收与输出

最大平移≤5mm、最大转角≤15°、包括闭合/稳定在内所有采样步接触穿透≤1mm，末尾连续1秒法向合力>0.001N且数值有效。acquire还需开重力前存在接触。任一步超限都不能用后续恢复抵消。

index.html有每候选原始下标、静态几何标记、六方向通过/失败/执行异常；链接是保存NPZ轨迹的碰撞网格回放，不重新做物理。results/status/REPORT随每方向完成更新；未完成不算失败。原static top_16 HTML与本批动力学页面不混淆。

plan.json包含物理参数、所有输入散列、非凸代理修补及选择身份；generation_surface.obj、collision_closed.obj、source_grasps.npy和ranking.json单独留存。动力学不检查全手语义/路径安全，不能称区域验收。

## 测试证据

CPU真实前16准备及已准备输入计划复核通过，16关节驱动/5mimic配置检查通过；2项排名回归测试通过（不按几何标记暗中筛选、能量不一致拒绝），Python编译检查通过。未新增GPU实验或完整96方向结果。
