# 电钻能量前16：Isaac Sim重新验证

日期：2026-11-13。此次重新运行原生Isaac Sim/PhysX，无RRL，不重生成姿态。

## 当前批次

目录：`data/experiments/power_drill_top16_isaacsim_sdf_20261113`。
日志：`data/experiments/power_drill_top16_isaacsim_sdf_20261113.log`。

选用和MuJoCo相同排名JSON及16个原始下标：11、18、20、23、0、10、4、31、28、1、6、8、24、22、27、5。比例0.2554999888，网格与参考来源散列核对，质量按相同密度500kg/m³和体积得到0.582463635kg。原始姿态不为获得成功而修改。

原生模型使用手部22个SDF与整电钻1个SDF（分辨率256），5个PhysX物理mimic、16个位置驱动；Kp5、Kd0.05、摩擦3。协议为零重力闭合2秒/稳定1秒、2400步×1/240秒重力保持，无物体位姿/速度重置。六方向各使用新进程，96次独立启动，避免跨方向求解器状态复用，墙钟耗时较长。

输入预检查通过：16下标与MuJoCo完全一致，质量也一致。首个方向运行时确认23三角网格形状、0凸形状、主动增益5/0.05并完成运行。此文记录启动快照，不宣称全部96方向已完成或成功；以summary_all.json及各方向实际结果为准。汇总以完整六方向候选计数，某候选方向进程已完成但尚未合并时不会计入完整候选数。

## 查看已有进度与结果

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
tail -n 20 data/experiments/power_drill_top16_isaacsim_sdf_20261113.log

# 一次性生成当前快照，非持续监控；批次结束也会自动生成。
.venv_l20/bin/python scripts/summarize_l20_isaacsim.py \
  --results data/experiments/power_drill_top16_isaacsim_sdf_20261113

xdg-open data/experiments/power_drill_top16_isaacsim_sdf_20261113/index.html
```

逐方向数据位于`power-drill/candidate_XXX/directions/gx+/`等目录；每方向有summary.json、SDF/驱动运行时审计和轨迹，候选合并后可在HTML回放切换六方向。index为保存轨迹可视化，不是再次运行物理。

## 重新启动新实验的命令

当前批次仍运行时不要重复执行。若需要另做新批次：

```bash
l20_sim_out="data/experiments/power_drill_top16_isaacsim_sdf_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/validate_ranked_isaacsim.py \
  --ranking data/experiments/l20_power_drill_original_size_20261113_151447/results/scales/scale_0p2555/top_16/ranking.json \
  --top-k 16 --output "$l20_sim_out"
```

可加`--plan-only`检查而不运行。任务异常退出后，可在相同命令加`--resume`审计复用完整候选；不完整候选的旧尝试归档后重新验证六方向。不要在原批次还运行时启动resume。

## 如何解释与MuJoCo的差异

保持了候选、几何、比例、密度、闭合时序、保持时间和主要位置增益/验收阈值，但没有声称只改变引擎：MuJoCo是1ms步长、SDF查询算法及软等式约束，PhysX是1/240秒、SDF256和PhysX mimic；接触求解与力指标也不同。MuJoCo首个严格失败提前结束，本Sim入口为完整运行后汇总整段最大值。

若Sim也失败，先比较同候选准备阶段穿透、开重力时接触与实际姿态；若Sim成功而MuJoCo失败，先分析模型局部变换、SDF距离/接触量及驱动差异，不能直接判某引擎准确或某姿态无效。仍保留0–6部分方向成功，不只报告6/6。
