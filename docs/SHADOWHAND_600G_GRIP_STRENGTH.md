# 600g电钻：抓取力与控制增益对照

2026-11-15。用户确认电钻质量600g，本批固定0.6kg（重力约5.886N），质量依据记为user_confirmed_600g；其他物理近似及原惯量值不改动。旧批次当时“假设质量”的历史措辞不回写成实测。

## 已有结果与原因证据

区域DexGraspNet原批次96方向已全部尝试，18通过、64物理失败、14非有限状态异常；status.complete=false包含异常，不是仍缺14个未运行方向。82完成中，64滑移超限、45角度超限、25穿透超限、22末尾接触不足（可重叠）；23方向首步已穿透超1mm，其余失败首个条件41次为滑移。候选14首步穿透约13.66mm，17约25.96mm，这类不能靠加力处理。

原控制Kp5/Kd0.05、力矩上限1Nm；根据旧轨迹关节位置差分估计qdot，候选30 gy+ PD需求峰值约0.573Nm、gz+约0.552Nm、gx+约0.326Nm，多数未达到上限。该估计不是PhysX隐式驱动实际输出，不能称已证实不饱和，但说明不能仅凭600g就认为1Nm上限必然是瓶颈。

位置控制初始target=q，没有主动额外预加载；增加力矩上限不会自动增加实际夹持力。提高刚度使同样偏差产生更大的保持力矩，同时需适当阻尼与穿透/数值检查。接触法向合力不等于抗重力合力或力矩平衡，不能只看总接触力数值。

## 独立版本与对照

- `validate_shadowhand_force_case.py`为从原生验证器复制的独立诊断版本，原验证器/RRL环境不改。记录来源代码SHA。
- `run_shadowhand_force_case.sh`沿用相同Isaac Sim入口与进程锁。
- `check_shadowhand_grip_strength.py`固定原候选30/原网格/初态/0.6kg/摩擦3/SDF256/10秒/阈值，三组各6方向、共18进程。

|组|Kp Nm/rad|Kd Nm·s/rad|关节上限Nm|
|---|---:|---:|---:|
|baseline|5|0.05|1|
|stiffer|20|0.1|1|
|stiffer_cap2|20|0.1|2|

前两组测试增益配置（刚度4倍、阻尼2倍，名义线性阻尼比保持）；后两组只改变力矩上限。所有组不修改参考角度、不新增闭合偏置、不改变初始穿透历史验收。是仿真对照，不是硬件额定扭矩声明。

每步保存240Hz实际关节速度、`estimated_pd_demand=kp*(target-q)-kd*qdot`及`solver_joint_efforts`。前者是后步状态估算，后者为求解器轴向载荷，不是隔离的电机力矩；SDK get_applied_joint_efforts主要反映set_joint_efforts命令，因此不把它当PD电机输出。get_max_efforts读回核对确实加载1/2Nm上限，估算超限比例仅作诊断。

## 执行状态

单方向gx-、1秒新版本接口检查已完成：数值有效、读回22个1Nm上限、诊断形状240×22且均有限。1秒通过不代表10秒稳定改善；该次位移3.147mm、穿透0.579mm、关节误差最大0.01268rad、PD估算峰值0.619Nm，不与旧10秒终值直接比较。

正式三组对照已启动：`data/experiments/dgn_drill_grip_strength_20261115`。不持续监控，不扩大到有严重干涉的候选。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
tail -n 12 data/experiments/dgn_drill_grip_strength_20261115.log
xdg-open data/experiments/dgn_drill_grip_strength_20261115/index.html
cat data/experiments/dgn_drill_grip_strength_20261115/REPORT.md
```

报告自动比较新fresh baseline，记录修复/损害方向、执行异常和通过数；轨迹/回放可查。需同时检查最大滑移/转角/穿透、关节运动与PD需求，不能只挑新增通过方向，也不能用求解器反力大于上限断定驱动饱和。

未来新批次（当前已运行，不要重复启动）：

```bash
force_out="data/experiments/dgn_drill_force_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/check_shadowhand_grip_strength.py \
  --baseline data/experiments/dgn_shadow_drill_regions_sdf_20261115 \
  --candidate 30 --output "$force_out" > "$force_out.log" 2>&1
```

通过判据仍是10秒平移≤5mm、旋转≤15°、全程穿透≤1mm、末尾连续1秒有效接触和数值正常。若加力增加穿透/振荡或破坏原成功方向，先回滚，再定位接触/目标偏置，不继续无界增力。
