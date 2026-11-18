# 杯子0：复核基线后开展残差修复

杯子20保留为工程回归参考；用户指定下一任务使用杯子原始下标0，scale=0.06。
每个reference独立训练策略，不加载杯子20或积木7检查点。

## 已知证据与目标边界

现有10秒acquire独立验证 `l20_all_acquire_10s_20261104_173323/mug/candidate_000/REPORT.md`
记录杯子0为4/6：gy+与gz+失败，其余四方向通过。
gy+最大滑移约410.8m、最大转角179.98°；这是无地面场景的长时间掉落记录，不能单凭有限大位移判为数值爆炸。
gz+最大滑移11.32mm、最大转角15.81°。历史direct报告为6/6，说明协议不可混用。
CPU计划检查确认当前v4可加载原始下标0、scale约0.06，但当前RL环境的失败标签及首次失败阶段尚未复测。
用户最新选择杯子0，取代此前杯子28计划；杯子28文档作为历史保留。

目标是：在当前RL环境确认基线失败后，策略修复失败方向，同时保留原通过方向及物理约束。
若基线全通过，先另设明确、两组一致的鲁棒性条件；不靠放宽验收或改变两组物理参数制造改善。

## 1. 当前RL环境零残差复核

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_mug0_base="data/experiments/mug0_rl_baseline_$(date +%Y%m%d_%H%M%S)"

bash scripts/validate_isaacsim_reference_baseline.sh \
  --object mug --candidate 0 \
  --num-envs 6 --episodes 1 \
  --position-noise 0.0005 --seed 20261109 \
  --output "$l20_mug0_base"

cat "$l20_mug0_base/REPORT.md"
cat "$l20_mug0_base/admission.json"
```

保持SDF、物理mimic、PD、摩擦、质量、2秒闭合/1秒稳定/10秒重力保持及严格验收不变。
一个±0.5mm平移复用于六方向；这是筛查，不是充分的鲁棒性统计。
`suitable=true` 表示准备阶段全部有效且存在重力保持失败，不保证动作来得及修复。
若全部通过，停止本次失败修复流程；若准备阶段失败，先处理参考或接触建立。

## 2. 仅在准入通过后，独立短训

沿用上一步shell变量；新终端须先把变量设为实际结果路径。

```bash
l20_mug0_train="data/experiments/mug0_residual_v4_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/train_isaacsim_reference.py \
  --baseline-results "$l20_mug0_base" \
  --num-envs 24 --iterations 200 --rollout-steps 32 \
  --output "$l20_mug0_train"
```

入口会拒绝不合适的基线。闭合阶段仍完全沿参考，只有重力阶段学习16关节残差。
200是采样批次数，实际有效更新及完整回合数须看训练记录。

## 3. 同样本配对，检查是否修复

```bash
l20_mug0_eval="data/experiments/mug0_residual_paired_$(date +%Y%m%d_%H%M%S)"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint "$l20_mug0_train/model_000200.pt" \
  --num-envs 6 --episodes 1 --position-noise 0.0005 \
  --baseline-results "$l20_mug0_base" \
  --samples "$l20_mug0_base/perturbations.json" \
  --output "$l20_mug0_eval"

cat "$l20_mug0_eval/REPORT.md"
xdg-open "$l20_mug0_eval/diagnostics.html"
```

先确认 `pass_flags_reproduced=true`，再看policy_only/zero_only、完整六方向通过、滑移/转角/穿透/mimic。
若确为两个重力失败方向，目标是二者均修复、原四个方向保持，并检查连续误差与关节运动代价。
若失败仍集中在重力开启后前100ms，分析限速目标、实际关节、接触力及首次失败时间，不能只增加训练轮数。
修复后使用固定检查点和新seed做配对复测；新样本不绑定旧单样本 `--baseline-results`。
进一步证明RRL价值，还需同条件简单控制器对照。

本流程尚未启动仿真或训练。原生Isaac Sim任务共用进程锁，已有任务会导致排队，不应强制停止已有任务。

## 首次当前RL复核结果与下一步

`data/experiments/mug0_rl_baseline_20261105_211756` 已完成，一个±0.5mm平移样本六方向全部通过，准备阶段也有效。
`no_baseline_failure_to_repair` 是正确拒绝，不应绕过准入。旧独立acquire的4/6未在该RL样本中复现；两者初态/克隆过程与求解器内部状态不同，尚未识别唯一原因。
下一步只扩大同范围样本，保持物理和验收不变，不训练：

```bash
l20_mug0_base5="data/experiments/mug0_rl_baseline5_$(date +%Y%m%d_%H%M%S)"
bash scripts/validate_isaacsim_reference_baseline.sh \
  --object mug --candidate 0 \
  --num-envs 6 --episodes 5 \
  --position-noise 0.0005 --seed 20261110 \
  --output "$l20_mug0_base5"
cat "$l20_mug0_base5/REPORT.md"
cat "$l20_mug0_base5/admission.json"
```

如仍全部通过，停止此条件下的失败修复；先对齐独立验证器与RL场景，再另设明确鲁棒性任务。
如准备阶段全部合格且存在重力失败，则用这个新目录训练；其同样本配对必须为 `--episodes 5`、读取新目录的perturbations，不能沿用单样本评估参数。

## 五样本复核已准入

`data/experiments/mug0_rl_baseline5_20261105_213034` 已完成：26/30，完整六方向2/5，准备有效。
4次重力滑移失败：组1 gy+、组3 gy+/gz+、组4 gy+；最早重力后33.3ms。
训练计划CPU检查通过，尚未运行训练。使用此目录作为 `--baseline-results` 独立短训200采样批次。
随后配对必须设置 `--num-envs 6 --episodes 5`、传入该目录的 `perturbations.json`，并确认基线标签复现。
首先验证原26次不受损以及4次失败能否修复，再评估新seed；不放宽验收，不自动扩大长训。

## 同样本短训结果与新seed复测

`mug0_residual_paired5_20261105_220153` 已完成，基线26/30、策略29/30，完整六方向2/5→4/5，修复3次、损害0次。
剩余组3 gy+重力后33.3ms滑移失败；滑移与mimic峰值误差中位数增大，不称全面改善。
固定模型，暂不继续训练：

```bash
l20_mug0_holdout="data/experiments/mug0_v4_holdout_$(date +%Y%m%d_%H%M%S)"
bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint data/experiments/mug0_residual_v4_20261105_214054/model_000200.pt \
  --num-envs 6 --episodes 5 --position-noise 0.0005 \
  --seed 20261111 --output "$l20_mug0_holdout"
cat "$l20_mug0_holdout/REPORT.md"
xdg-open "$l20_mug0_holdout/diagnostics.html"
```

这是新平移表，因此不传旧 `--baseline-results`、`--samples`、`--compare-results`。两组仍在同次评估共用新表。
新seed复测尚未运行；未依据当前结果扩大长训。

## 新seed复测结论

`mug0_v4_holdout_20261105_230459` 已完成：基线27/30、策略28/30，修复1次、损害0次，六方向均3/5。
两个gy+仍在重力后33.3/45.8ms失败，240Hz记录显示残差目标与关节已变化，但开启重力时总接触力弱或为零。
暂不继续长训。下一实验拟独立版本允许最后0.5s零重力稳定阶段接触预加载，先验证零动作等价、门控及接触响应；尚未实现，不跨载当前检查点。
证据与连续误差见工程复盘第26节。

## v5后续实验

v5已实现零重力稳定末段0.5秒的预加载残差，CPU/GPU短检查通过，未运行长训；完整流程见 [v5实验](ISAACSIM_RESIDUAL_V5.md)。保留当前v4，不跨载检查点。
