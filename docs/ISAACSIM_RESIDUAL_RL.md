# Isaac Sim：以原始抓取为参考的残差 PPO

本文保留v1流程。新实验入口见 [v2：历史状态与重力阶段学习](ISAACSIM_RESIDUAL_V2.md)，
v1脚本保持不变以兼容旧训练和检查点；v2观测与终止语义不同，需要新建策略。

本入口独立于旧 `mjlab_grasp`，直接加载当前六方向验证使用的 L20 SDF 场景。
**一份策略对应一个物体、一个比例、一个原始候选**；不是把五个物体混成同一策略，也不修改原始 NPY。
当前支持统一结果中的五物体和任意已完成、原始姿态合法的候选。

## 控制与任务

策略输出 16 维残差，实际目标为：

```text
requested_offset = 0.15 * tanh(action)       # rad，默认值
filtered_offset  = rate_limit(requested_offset, 0.2 rad/s)
target           = project_URDF_and_mimic(reference(t) + filtered_offset)
```

起始及回合重置时残差为零；连续零动作复现原参考轨迹。
从非零动作回到零动作时，残差会按限速逐渐归零，不瞬间跳回。
残差在闭合期间平滑启用；限速作用于目标偏置，不是实际关节速度的硬上限，完整参考闭合轨迹仍有自身速度。
投影同时考虑从动关节限位，主动关节的偏置速度也考虑 mimic 倍率。

- 参考来自原始生成 NPY 的同一候选：scale、手腕、21 个关节目标都保持对应关系，不重新缩放手腕。
- 拇指同样以原始抓取为参考，不沿用旧 mjlab 的自由拇指初态规则。
- 物理模型保留原 SDF 网格、质量、惯量、URDF 限位和力矩上限。
- 固定手腕，PD `kp=5`、`kd=0.05`，16 个驱动、5 个 PhysX **物理 mimic**，摩擦系数 3。
- 实际关节速度硬上限未启用，自碰撞关闭，与现有 SDF 验证设置一致。
- 物体自由，不添加支撑、焊接、抗重力补偿或抬升辅助力。

每回合同步进行：零重力闭合 **2 秒** → 零重力保持 **1 秒** → 重力验证 **10 秒**。
物理步长 1/240 秒，每四个物理步更新一次策略（60 Hz），每回合 780 个策略步。
环境使用六种场景整体旋转，在统一世界重力 -Z 下构造物体原始坐标中的六方向。
环境间通过碰撞组隔离；各环境阶段同步，因此真正切换场景重力，无需按环境施加补偿力。
掉落环境不提前重置，在本回合内停止获得正奖励，下一次同步回合再重置。

训练观测包含关节相对参考误差/速度、已滤波残差、物体位置/朝向误差、速度、重力方向、接触力、阶段及上一动作。
观测为仿真完整状态，当前没有部署到仅依赖实机传感器的策略。

奖励包括：近参考位姿、接触、低速度严格保持；惩罚残差幅度、动作变化、关节运动和穿透，掉落一次性扣分。
远离参考的物体不能仅凭停止运动继续获得保持奖励。奖励不是通过标准，仍须看配对评估。

## 检查计划

不启动 Isaac Sim、不创建输出目录：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

.venv_l20/bin/python scripts/run_isaacsim_residual.py \
  --mode plan --object mug --candidate 20 --num-envs 24 \
  --output data/experiments/isaacsim_residual_plan
```

默认 reference 目录为：
`data/experiments/l20_five_objects_acquire_10s_20261105`，可通过 `--reference-results` 更换。
必须是包含 `<物体>/summary.json` 的已完成结果目录，而非单独一个 NPY。
加载时核对源 NPY、URDF、网格及 USD，拒绝静默投影原始无效姿态。

## 短步检查

```bash
bash scripts/isaacsim_residual.sh \
  --mode smoke --object mug --candidate 20 \
  --num-envs 6 --steps 16 \
  --output "data/experiments/isaacsim_residual_smoke_$(date +%Y%m%d_%H%M%S)"
```

`num-envs` 必须为 6 的正整数倍，六方向均衡分配。
脚本使用 `/home/lee/isaac-sim/python.sh`；`L20_ISAACSIM_PATH` 可更改安装目录。
与验证/回放共用进程锁，避免同时启动多个本项目 Isaac Sim 实例。

## 训练 4000 次 PPO 更新

```bash
l20_rl_out="data/experiments/isaacsim_residual_mug20_$(date +%Y%m%d_%H%M%S)"

bash scripts/isaacsim_residual.sh \
  --mode train --object mug --candidate 20 \
  --num-envs 24 --iterations 4000 --rollout-steps 32 \
  --residual-scale 0.15 --residual-rate 0.2 \
  --position-noise 0.0005 \
  --output "$l20_rl_out"
```

每 100 次更新及最后一次保存检查点。此命令最终输出 `model_004000.pt`。
默认随机化仅包括回合初始物体位置每轴 ±0.5 mm，不随机质量、摩擦或控制增益。
参考候选即使历史上失败也能用于训练，但不是专家成功示范；小残差未必足以修正其几何缺陷。

更换参考可显式传入：

| 物体 | 命令中的参考示例 | 说明 |
|---|---|---|
| 杯子 | `--object mug --candidate 20` | 原始 NPY 下标 |
| 积木 | `--object jenga --candidate 7` | 原候选未通过全部六方向 |
| 香蕉 | `--object banana --candidate 13` | 独立训练策略 |
| 瓶子 | `--object bottle --candidate 2` | 独立训练策略 |
| 相机 | `--object camera --candidate 10` | 独立训练策略 |

以上不是把一个物体的检查点迁移给其他物体；检查点绑定 reference，跨参考加载会被拒绝。
每个物体的训练输出应使用新的目录。

## 评估：零残差与策略配对比较

在保存 `l20_rl_out` 的同一终端执行：

```bash
l20_rl_ckpt="$l20_rl_out/model_004000.pt"
l20_rl_eval="data/experiments/isaacsim_residual_eval_$(date +%Y%m%d_%H%M%S)"

bash scripts/isaacsim_residual.sh \
  --mode evaluate --checkpoint "$l20_rl_ckpt" \
  --num-envs 6 --episodes 1 \
  --output "$l20_rl_eval"

cat "$l20_rl_eval/REPORT.md"
```

评估自动恢复检查点对应的 reference 和控制配置，关闭初始位置扰动；先运行零残差，再运行确定性 actor。
两组都测完整三阶段、六个方向；不使用动作采样。
通过标准沿用当前 SDF 验证：获取接触，重力阶段最大位移≤5 mm、最大转角≤15°，
包括闭合阶段的采样接触穿透≤1 mm，最后连续1秒接触力>0.001 N，且数值有限。
额外记录实际 mimic 最大偏差作为诊断。

`all_six_directions_pass` 要求本次所有方向/重复试验均通过。
这是 RL 克隆场景中的配对评估，不替代原独立验证器；必须先检查零残差是否复现原候选的表现。
重复确定性初态不等于新的扰动鲁棒性验证，不应据此宣称已解决稳定性。

首批完整回合另存 `zero_residual_trajectory.npz`、`policy_trajectory.npz`，包含所有六方向的关节、目标、物体位姿和阶段。
评估记录用于新 PPO 策略，不与原 Gym/Isaac Sim 静态筛选结果混为一张成功率表。

## 在 Isaac Sim 窗口查看

```bash
bash scripts/isaacsim_residual.sh \
  --mode view --checkpoint "$l20_rl_ckpt" \
  --num-envs 6 --episodes 1 --view-env 5 \
  --output "data/experiments/isaacsim_residual_view_$(date +%Y%m%d_%H%M%S)"
```

窗口先运行零残差基线，再运行策略，每组13秒仿真时间；自动结束。
`--view-env 0..5` 对应 `gx+ gx- gy+ gy- gz+ gz-`；默认 5。
这是策略实时仿真，不是轨迹回放；鼠标可调整观察视角。

## 续训与曲线

续训使用新输出目录，`--iterations` 表示额外更新数：

```bash
bash scripts/isaacsim_residual.sh \
  --mode train --checkpoint "$l20_rl_ckpt" \
  --num-envs 24 --iterations 1000 --rollout-steps 32 \
  --output "data/experiments/isaacsim_residual_resume_$(date +%Y%m%d_%H%M%S)"
```

恢复模型、优化器和迭代编号；恢复后从新回合开始，不声称逐采样位的完全确定性续训。
参考与控制设置从检查点目录恢复。输出包括 `reference_snapshot.json`、内嵌几何的 `reference_scene.usda`、
`config.json`、`physics_audit.json`、`metrics.csv`、`model_*.pt` 和状态/异常记录。

### 已在训练：不重启，直接绘制 CSV 曲线

本机项目 `.venv_l20` 已有 TensorBoard，可将正在增长的 CSV 转成事件文件。
另开一个终端运行导出器（`--follow` 每5秒追加新指标，不修改训练或源CSV）：

```bash
.venv_l20/bin/python scripts/metrics_csv_to_tensorboard.py \
  --csv data/experiments/isaacsim_residual_mug20_20261105_121427/metrics.csv \
  --follow
```

再开一个终端启动 TensorBoard：

```bash
.venv_l20/bin/python -m tensorboard.main \
  --logdir data/experiments/isaacsim_residual_mug20_20261105_121427/tensorboard_csv \
  --host 127.0.0.1 --port 6006
```

打开 `http://127.0.0.1:6006`。关注 `train/mean_reward`、`train/value_loss`、
`evaluation/episode_six_direction_trial_pass_rate` 和 `train/action_std`。
`evaluation/all_six_directions_pass` 要求整批24个试验全部通过，不能当成单组六方向的通过率。
不加 `--follow` 时只导出当前快照；再次运行按已记录步骤增量导出，不重复追加已导出的指标。
导出器停止不影响训练，TensorBoard页面本身也不运行仿真。

### 后续新训练直接保存事件

本机 Isaac Sim Python 默认没有 TensorBoard。代码始终写入 `metrics.csv`；
如需事件曲线，可先在 Isaac Sim Python 中安装 TensorBoard（不是项目或 mjlab 环境）：

```bash
/home/lee/isaac-sim/python.sh -m pip install tensorboard
```

安装后新训练会额外保存事件文件，可通过 `bash scripts/tensorboard.sh` 查看。
数值异常不会静默复位或退回另一引擎，检查输出目录 `error.json`。

## 验证范围

CPU 检查覆盖：零动作 reference、限位/mimic 投影、残差速率、PPO 超时 bootstrap 和一次有限参数更新。
GPU 已跑通六环境短步 SDF/mimic 场景、一次 PPO 更新及检查点保存。
短测是代码链路检查，不构成训练后稳定性改善证据；未启动4000轮训练。

已完成的本机检查记录：

- `data/experiments/isaacsim_residual_smoke_v2_20261105/`：6环境，138个SDF运行时碰撞体，零动作参考检查。
- `data/experiments/isaacsim_residual_traincheck_20261105/`：仅1次PPO更新及检查点保存，不是训练完成的策略。
- `data/experiments/isaacsim_residual_evalcheck_20261105/REPORT.md`：杯子候选20，零残差/单次更新策略各六方向，完整13秒过程，均6/6。

配对短测复现了原参考的六方向通过，没有证明策略优于基线。
PhysX mimic 是求解器约束而非数学精确执行；短测的最大实际 mimic 误差分别约0.0024 rad（基线）与0.0115 rad（单次更新策略），
已记录供诊断。当前通过标准沿用原SDF验证，没有额外添加 mimic 误差阈值。

CPU 检查命令：

```bash
.venv_l20/bin/python -m pytest isaacsim_grasp/tests -q
```
