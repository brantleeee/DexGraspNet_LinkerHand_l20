# 杯子28：复核基线后开展残差修复

杯子20保留为工程回归参考；用户指定下一任务使用杯子原始下标28，scale=0.06。
每个reference独立训练策略，不加载杯子20或积木7检查点。

## 已知证据与目标边界

现有 `l20_isaacsim_sdf_all_20261104_1529/REPORT.md`（direct）与
`l20_all_acquire_10s_20261104_173323/mug/candidate_028/REPORT.md`（acquire、重力10秒）
均记录杯子28为6/6。后者最大滑移1.46mm，最大转角9.15°。
因此“已有两个失败方向”尚未与当前资产和协议对应，不能指定或虚构其方向。
CPU计划检查确认当前v4可加载该reference，但尚未运行它的RL环境动力学复测。

目标是：在当前RL环境确认基线失败后，策略修复失败方向，同时保留原通过方向及物理约束。
若基线全通过，先另设明确、两组一致的鲁棒性条件；不靠放宽验收或改变两组物理参数制造改善。

## 1. 当前RL环境零残差复核

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_mug28_base="data/experiments/mug28_rl_baseline_$(date +%Y%m%d_%H%M%S)"

bash scripts/validate_isaacsim_reference_baseline.sh \
  --object mug --candidate 28 \
  --num-envs 6 --episodes 1 \
  --position-noise 0.0005 --seed 20261109 \
  --output "$l20_mug28_base"

cat "$l20_mug28_base/REPORT.md"
cat "$l20_mug28_base/admission.json"
```

保持SDF、物理mimic、PD、摩擦、质量、2秒闭合/1秒稳定/10秒重力保持及严格验收不变。
一个±0.5mm平移复用于六方向；这是筛查，不是充分的鲁棒性统计。
`suitable=true` 表示准备阶段全部有效且存在重力保持失败，不保证动作来得及修复。
若全部通过，停止本次失败修复流程；若准备阶段失败，先处理参考或接触建立。

## 2. 仅在准入通过后，独立短训

沿用上一步shell变量；新终端须先把变量设为实际结果路径。

```bash
l20_mug28_train="data/experiments/mug28_residual_v4_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/train_isaacsim_reference.py \
  --baseline-results "$l20_mug28_base" \
  --num-envs 24 --iterations 200 --rollout-steps 32 \
  --output "$l20_mug28_train"
```

入口会拒绝不合适的基线。闭合阶段仍完全沿参考，只有重力阶段学习16关节残差。
200是采样批次数，实际有效更新及完整回合数须看训练记录。

## 3. 同样本配对，检查是否修复

```bash
l20_mug28_eval="data/experiments/mug28_residual_paired_$(date +%Y%m%d_%H%M%S)"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint "$l20_mug28_train/model_000200.pt" \
  --num-envs 6 --episodes 1 --position-noise 0.0005 \
  --baseline-results "$l20_mug28_base" \
  --samples "$l20_mug28_base/perturbations.json" \
  --output "$l20_mug28_eval"

cat "$l20_mug28_eval/REPORT.md"
xdg-open "$l20_mug28_eval/diagnostics.html"
```

先确认 `pass_flags_reproduced=true`，再看policy_only/zero_only、完整六方向通过、滑移/转角/穿透/mimic。
若确为两个重力失败方向，目标是二者均修复、原四个方向保持，并检查连续误差与关节运动代价。
若失败仍集中在重力开启后前100ms，分析限速目标、实际关节、接触力及首次失败时间，不能只增加训练轮数。
修复后使用固定检查点和新seed做配对复测；新样本不绑定旧单样本 `--baseline-results`。
进一步证明RRL价值，还需同条件简单控制器对照。

本流程尚未启动仿真或训练。原生Isaac Sim任务共用进程锁，已有任务会导致排队，不应强制停止已有任务。
