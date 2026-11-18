# 残差 PPO v2：可观察的严格历史、只学习重力保持

v2 单独使用 `scripts/isaacsim_residual_v2.sh` / `scripts/run_isaacsim_residual_v2.py`。
**v1 文件保持不变**，旧训练无需重启，原检查点仍通过原入口评估。
v2 改变了观测和任务语义，必须新建策略；不能直接加载 v1 检查点或覆盖旧目录。

## 本次修改范围

保留 SDF、固定 PD 5/0.05、物理 mimic、原始 reference、2秒闭合+1秒零重力保持+10秒重力验证。
本次没有修改残差幅度/速率、探索噪声、学习率或 PPO 更新公式。

### 1. 历史与时间状态进入观测

观测由93维增加为120维，增加27项：

- 回合已用/剩余比例、闭合/零重力保持/重力阶段进度。
- 最大滑移、最大转角、最大穿透及当前穿透。
- 连续接触时长（达到1秒后饱和）、当前接触点数量。
- 是否已检查获取、是否获取接触、是否严格失败、是否掉落。
- 滑移、转角、穿透、获取及末秒连续接触五类失败标记。
- 开启重力时捕获的参考物体位置和四元数。

历史每个物理步更新，避免只检查60Hz策略步而漏掉中间瞬时越界。

### 2. 奖励、逻辑终止与验收使用同一历史

严格阈值保持不变：最大滑移≤5mm，最大转角≤15°，全程采样接触穿透≤1mm；
开启重力前已获取接触，最后连续1秒接触力>0.001N。

超过不可恢复的阈值后：

- 标记严格失败，首次失败扣5分。
- 后续不能通过回到原位再次领取保持正奖励。
- 冻结当时残差目标，不继续探索；失败后状态不参与PPO更新。
- 该失败转换是终止转换，不从失败状态的价值预测进行bootstrap。

接触暂时丢失不是立即失败：早期丢失后仍可重新接触，只要剩余时间足以填满最后1秒窗口。
进入最后窗口后，只有当连续接触时间加剩余时间不足1秒，才标记不可恢复失败。
全程超标后恢复、闭合阶段穿透超标、未获取接触，都不能在末帧冒充成功。
只有最终完整通过才领取额外5分成功奖励。

固定13秒任务结束时，成功或失败都视作真正终止，不再对终点做无限时长超时bootstrap。
为保持所有环境的真实重力切换同步，逻辑失败后场景仍运行到统一回合末尾，再统一复位，
而不是立即单独复位一个环境。数值非有限仍直接报错，不静默掩盖。

### 3. 只学习重力阶段残差

闭合和零重力保持时，不论策略输出多大：

- 完全执行原参考目标。
- 残差滤波状态和上一有效动作保持为零。
- 不向actor或critic提交这些阶段的训练样本，也不进行熵更新。

开启重力后，从零残差开始执行原有幅度/速率限制，仅仍可通过验收的环境参与学习。
重力阶段内某步首次严格失败，该转换仍参与训练，后续吸收状态排除。
失败事件在每个物理子步检查；若一个策略步中途失败，后续子步立即冻结目标。

`--iterations` 在v2中表示采样批次数，不保证每一批都执行PPO更新。
`train/eligible_gravity_samples`、`train/optimizer_updates` 和 `train/update_performed` 显示实际学习情况。
纯闭合或全为已失败环境的采样批次会跳过更新；控制台显示 `value_loss=skipped`，不是损失收敛为零。
`train/gravity_reward` 只统计有效重力样本；`train/mean_reward` 是整批采样均值，包含零奖励准备阶段。

## 使用命令

先检查计划，不启动GPU：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

.venv_l20/bin/python scripts/run_isaacsim_residual_v2.py \
  --mode plan --object mug --candidate 20 --num-envs 24 \
  --output data/experiments/isaacsim_residual_v2_plan
```

当前v1进程退出并释放共享Isaac Sim锁后，做短步检查。200个策略步会经过闭合/保持并进入重力阶段：

```bash
bash scripts/isaacsim_residual_v2.sh \
  --mode smoke --object mug --candidate 20 \
  --num-envs 6 --steps 200 \
  --output "data/experiments/isaacsim_residual_v2_smoke_$(date +%Y%m%d_%H%M%S)"
```

先新建200批次短训，保持原残差和学习率，只验证本次两项改动：

```bash
l20_v2_out="data/experiments/isaacsim_residual_v2_mug20_$(date +%Y%m%d_%H%M%S)"

bash scripts/isaacsim_residual_v2.sh \
  --mode train --object mug --candidate 20 \
  --num-envs 24 --iterations 200 --rollout-steps 32 \
  --residual-scale 0.15 --residual-rate 0.2 --position-noise 0.0005 \
  --output "$l20_v2_out"
```

评估仍配对运行零残差与确定性策略，关闭初始位置扰动：

```bash
bash scripts/isaacsim_residual_v2.sh \
  --mode evaluate --checkpoint "$l20_v2_out/model_000200.pt" \
  --num-envs 6 --episodes 1 \
  --output "data/experiments/isaacsim_residual_v2_eval_$(date +%Y%m%d_%H%M%S)"
```

当前120维观测、历史和终止语义被纳入版本与代码散列检查。
输入依旧是原五物体结果目录，其他物体只需替换物体别名和原始候选下标。
若参考的闭合阶段本身不满足严格标准，重力阶段策略无法补救已经失败的闭合；需先改善参考或另选候选。

## 检查范围

CPU测试覆盖：历史可观察、瞬时超标后恢复仍失败、早期接触丢失可恢复、末秒丢失不可恢复、
闭合穿透/获取失败、有限时长终止不bootstrap、非零闭合动作被忽略、重力动作启用、失败目标冻结、mimic投影。
已检查原v1训练配置的全部代码散列仍匹配，因此本次改动没有破坏旧入口。

本次没有停止或修改现有训练，也未启动v2 GPU仿真或长训。真实SDF物理行为和训练效果仍需通过上述短步检查及评估确认。

## 配对初始位置扰动评估（已有v2检查点可直接使用）

独立入口 `scripts/evaluate_isaacsim_residual_paired.sh` 不修改v2训练代码，也不绕过检查点散列检查。
原入口 `--mode evaluate` 仍为无扰动评估；需要扰动时使用本节新入口。

只引入每轴独立均匀采样的初始物体平移 **±0.0005 m（±0.5 mm）**。
不随机旋转、质量、摩擦、PD、力矩限制、mimic、物理步长或三阶段时长。
其余设置从检查点继承；`num-envs` 只改变并行环境数，不改变每条试验的物理设置。

先生成明确的样本表，而不是仅依赖两组重新设置相同seed：

- 每个扰动平移在物体原始坐标中定义，同一平移测试六个重力方向。
- `zero_residual` 和确定性 `policy` 逐项读取同一个 `perturbations.json`。
- 显式复位手腕根位姿/速度、关节、残差滤波器、物体位姿/速度和历史状态。
- 保存 `initial_states.npz`，对两组的显式初态逐项检查完全一致；不一致则停止，不静默容忍。
- 请求位置到PhysX读回值允许浮点量化误差（最大2微米），两组之间的显式初态仍要求逐项相等。
- 固定显式初态不等于清除PhysX所有内部warm-start状态，也不保证后续轨迹逐位确定。

先检查计划（无GPU、无输出目录创建）：

```bash
.venv_l20/bin/python scripts/evaluate_isaacsim_residual_paired.py \
  --checkpoint data/experiments/residual_v2_mug20_20261105_144025/model_000200.pt \
  --num-envs 6 --episodes 5 --position-noise 0.0005 --seed 20261105 \
  --plan-only --output /tmp/l20_v2_paired_plan
```

正式运行（由用户启动；与其他Isaac Sim任务共用锁）：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_pair_out="data/experiments/residual_v2_paired_0p5mm_$(date +%Y%m%d_%H%M%S)"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint data/experiments/residual_v2_mug20_20261105_144025/model_000200.pt \
  --num-envs 6 --episodes 5 \
  --position-noise 0.0005 --seed 20261105 \
  --output "$l20_pair_out"

cat "$l20_pair_out/REPORT.md"
```

本命令使用5个平移样本，每个测试六方向：每组30次方向试验，基线和策略总计60次。
默认评估seed不同于训练seed42，幅度与训练相同；这是同分布新样本筛查，不是重放训练样本。
5个样本不足以证明广泛鲁棒性，先看方向级差异与约束，再考虑扩大样本。

重点检查：

- `policy_only`：策略通过、零残差失败，表示这个配对试验有收益。
- `zero_only`：零残差通过、策略失败，表示这个配对试验出现能力损失。
- `both_pass` / `both_fail`：两者一致的试验。
- 每方向通过数，以及每个平移的六方向全通过数。
- `trials.csv` 中滑移、转角、穿透、实际mimic误差和失败原因，避免只看总通过率。

输出文件：

- `plan.json` / `config.json`：训练/评估配置及允许的差异。
- `perturbations.json`：保存的实际请求样本和seed，可复用给其他同参考检查点。
- `initial_states.npz`：两组的显式物理及控制器初始状态。
- `evaluation.json`：全部试验、初态核对结果、配对统计。
- `trials.csv` / `REPORT.md`：可直接比较的结果。
- `physics_audit.json` / `evaluator_sources.json`：SDF/驱动/mimic审计及评估代码散列。

复用样本时加 `--samples "$l20_pair_out/perturbations.json"`，同时保持episodes、num-envs和position-noise一致；
样本文件的seed为准。只有输出目录需要更换。

检查范围：6项CPU配对测试通过；本机原生Isaac Sim的两批初态复位检查通过，
两组显式状态差异为0，请求位置最大读回误差约7.61e-8 m。该检查没有推进物理步，
未代用户执行正式扰动稳定性评估；`--check-resets-only` 输出不能作为通过率结果。
