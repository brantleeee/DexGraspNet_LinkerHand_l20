# 电钻排名前16：MuJoCo原生SDF动力学验证

日期：2026-11-13。直接调用MuJoCo，不使用mjlab或训练策略。

同批前16已另启动[Isaac Sim SDF复核](POWER_DRILL_ISAACSIM_SDF.md)，保持原候选/尺度/网格及质量假设，不覆盖本MuJoCo结果。

本次已启动完整16×6筛选：`data/experiments/power_drill_top16_mujoco_sdf_1ms_20261113`；日志为同名`.log`文件。此文记录实现与启动，不宣称该批次已全部完成；以results.json的complete字段和最终REPORT.md为准。早期`power_drill_top16_mujoco_sdf_20261113`为较软联动的初筛，不能代替最终轮。

**后续完成快照**：96/96执行完成、严格通过0/96；84准备阶段失败、12重力阶段失败，mimic最大偏差0.002945rad。准备失败为60次穿透、24次未获取接触。排名4/6进入重力但均未通过。静态接触短诊断显示排名1原闭合参考在MuJoCo即有约10.43mm穿透记录，需核对生成/导入几何变换及不同穿透度量，不能仅归因闭合轨迹或重力。结果不证明原候选在其他控制/协议下全不可用。详细见工程复盘第55节。

## 选择与尺寸

输入是原尺寸生成目录的 `results/scales/scale_0p2555/top_16/ranking.json`，按该文件中的真实候选下标取能量前16，不是取数组0–15。源NPY SHA256、候选能量、比例与URDF可行性均检查；几何FAIL也保留验证，不隐式换成其他候选。

物体最长边约255.5mm，沿生成scale=0.2554999888，不重复缩放手腕。采用已修复非凸网格，不做凸包或凸分解；手部22个原始网格加1个整物体网格全部设置原生SDF，并在编译后检查23个碰撞geom均为SDF。5个mimic等式、16个独立位置驱动，不能仅靠目标联动代替实际耦合。

## 验证协议

- 固定手腕，无物体焊接、支撑平面或外部保持力。
- 0–2秒零重力平滑闭合（屈曲朝中性张开60%再闭合）；2–3秒稳定；3秒后指定方向9.81m/s²重力，成功方向保持10秒。
- 六方向相对于物体导入初始模型轴，MuJoCo直接设置该坐标系重力，不按数组下标猜测方向。
- Kp=5、Kd=0.05，驱动±1Nm；另有被动关节阻尼0.02；摩擦参数3/0.005/0.0001，自碰撞关闭，Newton求解器100次迭代。
- 默认物理步长1ms；mimic等式solref=0.002/1、solimp=0.999/0.999/0.001/0.5/2；SDF起始点40、迭代10。实际约束误差另报告，等式并非数学刚体硬锁。
- 物体质量由封闭网格体积×500kg/m³推算，约0.58246kg，不是称重值，惯量也由该网格体积分布计算。

严格条件：3秒获取接触记录数>0；重力最大滑移≤5mm、转角≤15°；全程采样穿透≤1mm；末1秒连续法向接触力指标>0.001N；数值有效。失败历史不可逆。MuJoCo力指标为各手–物接触绝对法向力之和，与Isaac Sim的逐连杆法向合力模长求和并不完全相同，所以不是同参数引擎准确性排名。

失败方向在首次严格失败后结束，保留前缀轨迹与实际时长；这不等于所有方向都跑满13秒。通过方向必须完成全部阶段和末1秒接触。`mimic_quality_pass`仅为≤0.02rad的额外诊断，不能忽略大偏差后宣称机构执行可信。

## 运行命令

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

l20_mj_out="data/experiments/power_drill_top16_mujoco_sdf_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/validate_l20_mujoco_sdf.py \
  --ranking data/experiments/l20_power_drill_original_size_20261113_151447/results/scales/scale_0p2555/top_16/ranking.json \
  --top-k 16 \
  --gravity-seconds 10 \
  --timestep 0.001 \
  --workers 2 \
  --output "$l20_mj_out"

xdg-open "$l20_mj_out/index.html"
```

可先加`--plan-only`核对排名/尺寸，不运行物理。输出目录必须为新目录。初版4.167ms试验保存为历史初筛，存在较大的软联动偏差；应以1ms及加强约束的独立结果为准，不混合统计。

输出包括index.html、REPORT.md、results.json、collision_audit.json、plan.json，及每候选scene.xml/scene.mjb、每方向result.json/trajectory.npz。报告区分数值/执行异常、准备阶段失败和重力阶段失败。

重建已有完整结果的页面，无需重新验证：

```bash
.venv_l20/bin/python scripts/validate_l20_mujoco_sdf.py \
  --output "$l20_mj_out" --summarize-only
```
