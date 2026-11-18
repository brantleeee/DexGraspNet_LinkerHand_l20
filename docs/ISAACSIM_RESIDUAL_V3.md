# v3：保留零残差基准，优化稳定裕量并诊断动作

本轮目标是保持严格通过能力，同时减小滑移、转角和不必要的残差。
零残差仍是默认对照，不因奖励更高或策略动作更明显而替换它。
原v1/v2训练代码和检查点散列保持不变；v3需新建策略，不能把v2优化器/价值网络静默续训到新奖励。

## 改变与保持

只改变奖励目标，保留v2的120维观测、重力阶段学习、严格历史终止、13秒过程、
SDF/物理mimic、PD 5/0.05、摩擦、质量/惯量、限位/力矩上限，以及残差幅度0.15rad、偏置限速0.2rad/s。

原奖励的位姿尺度较宽，且阈值内稳定保持项近似相同。
v3使用连续的1mm位置、2°转角尺度，奖励随误差连续下降；线/角速度尺度为0.02m/s、0.3rad/s。
增加对历史误差、实际目标残差、有效请求残差和残差变化的惩罚。
具体权重保存在 `isaacsim_grasp/margin_reward.py` 和每次新训练的 `config.json`。
其中0.05rad是残差代价的归一化尺度，不是将动作硬限幅改成±0.05rad。

严格通过阈值没有改变。纯零残差、低误差可以获得高奖励，不要求策略为展示能力而移动手指。
奖励尺度改变后，v2/v3的奖励数值不能直接比较，需比较同样本的通过数及连续误差。
这些是待验证的工程选择，不保证一定改善。

## 动作诊断

配对评估现在对v2/v3都以240Hz记录：

- `raw_action`：actor原始16维输出，无量纲；准备阶段也可非零，但被环境屏蔽。
- `requested_residual`：`scale * tanh(raw_action)`，rad。
- `effective_requested_residual`：考虑准备阶段/失败屏蔽的请求，rad。
- `applied_residual`：实际关节目标减reference，仅16个主动关节，已经过限速、限位/mimic投影及失败冻结，rad。
- 原有21关节实际位置、速度、目标跟踪、接触、穿透和力矩数据继续保留。

请求、实际目标、实际运动是三层不同的数据；目标限速不等于实际关节速度上限。
旧数据没有保存的原始动作不能从最终关节位置反推，需要重新运行已保存检查点来记录。
评估中的read-only诊断不向物体加力，也不修改控制设置。

`diagnostics.html` 在原统一位置/速度/力矩图后增加四个动作面板；
`evaluation.json` 每项试验的 `action_summary` 包括RMS、峰值、每关节均值、实际目标偏置和关节运动路程。
偏置与RMS按整个重力阶段计算，`enabled_gravity_frames` 标记有效控制帧数；失败后冻结偏置仍会被记录。
这些量用于判断持续偏置，不自动证明所有非零残差都没有必要。

训练只保存采样批次的动作/奖励分项标量到 `metrics.csv`，避免全量240Hz训练轨迹消耗磁盘。
新增标签包括 `action/applied_residual_rms_rad`、`action/applied_residual_abs_bias_rad`、
`action/requested_residual_rms_rad`、`action/joint_velocity_rms_rad_s` 及 `reward/*`。
可继续使用 `scripts/metrics_csv_to_tensorboard.py --csv <目录>/metrics.csv --follow` 查看。

## 先诊断已有v2策略（可选，无需重新训练）

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_samples="data/experiments/residual_v2_paired_0p5mm_20261105_153925/perturbations.json"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint data/experiments/residual_v2_mug20_20261105_144025/model_000200.pt \
  --num-envs 6 --episodes 5 --position-noise 0.0005 \
  --samples "$l20_samples" \
  --output "data/experiments/residual_v2_action_diagnostics_$(date +%Y%m%d_%H%M%S)"
```

完成后打开该目录 `diagnostics.html`。准备阶段的原始actor输出非零不代表目标改变，
应同时看有效请求和实际目标残差。

## 只做200批次v3短训

```bash
l20_v3_out="data/experiments/residual_v3_mug20_$(date +%Y%m%d_%H%M%S)"

bash scripts/isaacsim_residual_v3.sh \
  --mode train --object mug --candidate 20 \
  --num-envs 24 --iterations 200 --rollout-steps 32 \
  --residual-scale 0.15 --residual-rate 0.2 --position-noise 0.0005 \
  --output "$l20_v3_out"
```

不传v2检查点；v3初始确定性actor为零。采样批次和实际PPO更新数仍分开记录。

## 复用原样本，与v2比较

```bash
l20_v3_eval="data/experiments/residual_v3_paired_$(date +%Y%m%d_%H%M%S)"
l20_samples="data/experiments/residual_v2_paired_0p5mm_20261105_153925/perturbations.json"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint "$l20_v3_out/model_000200.pt" \
  --num-envs 6 --episodes 5 --position-noise 0.0005 \
  --samples "$l20_samples" \
  --compare-results data/experiments/residual_v2_paired_0p5mm_20261105_153925 \
  --output "$l20_v3_eval"

cat "$l20_v3_eval/REPORT.md"
cat "$l20_v3_eval/MARGIN_COMPARISON.md"
xdg-open "$l20_v3_eval/diagnostics.html"
```

评估开始前核对旧结果完整、样本相同、物理/控制设置相同，完成后输出：

- `REPORT.md`、`trials.csv`、`evaluation.json`：本次零残差/策略通过情况及动作统计。
- `MARGIN_COMPARISON.md`、`margin_comparison.json`：旧v2与新v3各自基线/策略的配对连续误差比较。
- `diagnostics/*.npz`、`diagnostics.html`、`diagnostics_data/`：240Hz动作和物理数据及离线图。

也可单独比较两个已经完成的同样本结果，无需再仿真：

```bash
.venv_l20/bin/python scripts/compare_isaacsim_margin.py \
  --results "$l20_v3_eval" \
  --previous data/experiments/residual_v2_paired_0p5mm_20261105_153925
```

判断顺序：先确保通过数不下降和物理约束未恶化，再看滑移/转角的配对误差、最坏值及残差偏置。
两次零残差若明显不一致，应先排查环境复现，不能直接把差异归因于奖励。
当前只有5个平移样本；观察到改善后再扩大覆盖，不直接扩大长训。

## 已做检查

- CPU行为测试验证奖励能区分阈值内误差，非零残差有代价，失败不能重新领分。
- 相同动作下v2/v3的观测、目标命令和验收结果一致；只有奖励改变。
- 动作诊断测试区分原始动作、请求、实际目标和实际关节运动，保留原力矩语义。
- 比较器校验相同样本/物理配置，并能在两组全通过时报告连续误差改善/恶化。
- 原生Isaac Sim以6环境运行1个184步采样批次，越过重力切换、完成一次PPO更新及检查点保存。
- 实际短测v3检查点已通过配对评估计划加载、原样本复用与物理参数预检查。

未启动200批次训练或正式扰动稳定性复测，当前没有v3策略优于零残差或v2的证据。
