# GraspQP 原生ShadowHand：电钻全关节生成与Isaac Sim验证

当前任务目录：`data/experiments/graspqp_shadow_drill_sdf_20261115`；日志：同名`.log`。任务已启动，生成32候选、优化3000步，再按能量前16做物理验证。没有训练神经网络，也不复用GraspGenX近似描述或固定手指模板。

## 本地方案与适配

源项目`/home/lee/robot_learning/open_source_baselines/graspqp`，环境`/home/lee/miniconda3/envs/graspqp/bin/python`。使用自带`shadow_hand`URDF、接触定义、WARP SDF、GraspQP QP力闭合目标与MALA*优化器，优化基座位姿和全部22活动关节。原工厂说明写24，但实际URDF固定了两个腕关节。原生默认状态仍保留这两项，调用层明确去掉；不改变外部文件。

本地GraspQP环境的pytorch_kinematics fork会给每个旋转关节角隐式增加±0.01rad。初次导出FK发现约0.0336矩阵误差并停止，未将该结果用于正式验证。新流程在生成/导出各进程中使用精确旋转角实现，保留自动微分，不修改外部库。正/负/零角及零角梯度断言通过。

物体用同一已修复的封闭非凸电钻，预先恢复源数值尺寸255.5×223.4×75mm保存为物理单位网格，GraspQP scale=1。不是把原归一化网格当作1米电钻。进入验证器后也保持scale=1、同一物理网格和手腕坐标。体积×500kg/m³估算质量约0.582464kg，非实测重量。

所有手部资源先复制到本批目录，接触缓存只写本工程。原URDF惯量为零：显式以原质量和碰撞几何AABB箱体补正对角惯量，保留源质心；无惯性固定帧使用0.1g质量下限。所有补正逐连杆记录`inertia_repairs`。这是可运行近似动力学模型，不是实测ShadowHand标定；不能据通过率声称真实硬件更好。

## 验证协议

- 每个合法候选每方向独立Isaac Sim进程，所有手部非空碰撞几何及电钻均SDF256，无凸化；初始位姿直接加载，固定手部基座，关节目标为各候选自己的优化角度，无额外闭合修正。
- 六个初始模型坐标方向各10秒，物体自由，无支撑或焊接。时间步1/240秒、Kp5/Kd0.05、单关节力矩上限1Nm（源URDF effort=0另作显式验证配置）、接触摩擦3。原GraspQP优化摩擦参数仍为默认0.2，优化参数与物理接触系数不能混为同一项。
- 严格保持最大平移≤5mm、转角≤15°、穿透≤1mm、末尾连续1秒法向合力>0.001N且数值有效。每个越界候选明确拒绝，不截断角度，记入`input_audit.json`；不会用更低排名候选补位后仍称原前16全通过。
- 本流程验证静态生成姿态的保持能力，不证明从张开状态能抓取。不是GraspQP官方Isaac Lab验收标准的复现，而是本工程原生SDF协议。

## 已完成短检查

`graspqp_drill_smoke_v3_20261115`：2候选×2步仅用于链路检查；2候选输入合法，生成器/URDF FK最大误差约4.46e-7。单候选gz-、1秒原生Isaac Sim短检查完成，19个三角网格形状、0凸形状、22关节、增益5/0.05、数值有效；该短抓取未通过，不是稳定性改善证据。

## 查看一次快照

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
tail -n 20 data/experiments/graspqp_shadow_drill_sdf_20261115.log
```

生成和导出后：

```bash
xdg-open data/experiments/graspqp_shadow_drill_sdf_20261115/generation_index.html
xdg-open data/experiments/graspqp_shadow_drill_sdf_20261115/index.html
```

`generation_index.html`为静态生成姿态，`index.html`为逐候选逐方向状态，链接HTML为实际保存轨迹回放，非新物理仿真。尚未出现文件表示仍未进入对应阶段，不能当作失败。

## 新批次完整命令（当前已有任务，不要重复启动）

```bash
qp_out="data/experiments/graspqp_shadow_drill_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/graspqp_drill_pipeline.py pipeline \
  --batch-size 32 --n-iter 3000 --top-k 16 --output "$qp_out" \
  > "$qp_out.log" 2>&1
```

原生进程共用现有Isaac Sim锁，其他任务占锁时排队。本版无自动续跑，旧输出拒绝覆盖。分阶段`generate/export/render/validate`入口用于定位失败；不得对已含试验目录直接重新运行validate。
