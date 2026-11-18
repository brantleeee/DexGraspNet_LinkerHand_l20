# v5评估专用预加载倍率消融

2026-11-06。训练实现与检查点不变，只改变确定性评估时的预加载请求。

## 实验合同

- 已有原倍率1结果：`mug0_v5_second_deck_20261106_103919`，基线27/30、策略27/30，完整六方向3/5→2/5，修复2次/损害2次。
- 新参数 `--preload-gain 0.5` 仅在2.5–3秒对策略的tanh后请求残差乘0.5。
- 在3秒后不缩放策略请求；旧控制器的幅度/速率约束与偏置记忆保持，切换时不清零或强制跳目标。
- 这不是把raw action乘0.5，也不是把全局residual-scale减半；实际关节运动不会被强制减半。
- 零残差控制不施加此干预，两个控制仍共用同一份初始扰动；物理、奖励、时长和验收不变。
- 半倍率是同一检查点的评估控制消融，不是新训练或原策略原样输出。plan/evaluation/report会明确记录干预。

## 执行命令

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_half_out="data/experiments/mug0_v5_half_preload_$(date +%Y%m%d_%H%M%S)"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint data/experiments/mug0_residual_v5_20261106_100328/model_000200.pt \
  --num-envs 6 --episodes 5 --position-noise 0.0005 \
  --preload-gain 0.5 \
  --samples data/experiments/mug0_v5_second_deck_20261106_103919/perturbations.json \
  --compare-results data/experiments/mug0_v5_second_deck_20261106_103919 \
  --output "$l20_half_out"

cat "$l20_half_out/REPORT.md"
cat "$l20_half_out/MARGIN_COMPARISON.md"
xdg-open "$l20_half_out/diagnostics.html"
```

不传第一批baseline-results；本次与第二批原倍率结果比较。已有Isaac Sim任务会排队。

## 先判断什么

1. 先确认跨次零残差标签和连续误差复现。
2. 新增损害组0 gy-、组4 gx-能否恢复？这两例此前预加载推移约5.80/12.26mm且开重力总力0。
3. 两个修复（组2 gy+、gz+）能否保留？看policy_only和zero_only，不能只看净通过数。
4. 完整六方向能否恢复至至少基线3/5，再看滑移/转角/穿透/mimic及预加载推移。

若损害消失且修复保留，支持减小预加载幅度的方案，仍需其他样本确认。
若损害和修复同时消失，说明存在控制幅度取舍，不能称鲁棒改进。
若损害不变，优先查逐指接触位置、法向、力及动作方向，而不是继续缩幅或延长训练。

## 诊断与已做验证

新增 `isaacsim_grasp/evaluation_interventions.py`，不修改任何已训练env/control/reward/PPO代码。
NPZ字段：policy_raw_action/policy_requested_residual为原actor输出，raw_action/requested_residual为送入环境的等效命令，preload_request_gain为当帧倍率，applied_residual为限速与投影后的实际目标偏置。
原诊断大图的raw/requested显示送入环境的命令；原actor值保存在新增NPZ字段，不能混读。

CPU全套38项通过，覆盖tanh后缩放、零/半/全倍率、前缀与重力阶段不缩放、无效倍率拒绝、跨重力偏置/限速连续性，以及双层动作记录。
当前检查点训练散列保持一致，真实半倍率命令CPU plan-only通过。
本轮未启动GPU评估或训练；半倍率是否改善稳定性待验证。

## 第二批半倍率结果与第一批交叉复测

第二批半倍率已完成：`mug0_v5_half_preload_20261106_115332`，基线27/30、策略29/30，完整六方向3/5→4/5，修复2/损害0。
原全倍率两个新增损害已恢复、两个修复保留，但剩余组3 gy+未修复。mimic最大0.04695rad发生在2.06667秒的index_dip，早于倍率干预，不直接归因于半倍率。
下一步冻结倍率0.5，回第一批相同样本检查是否保留原修复，不搜索新倍率：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_half_first="data/experiments/mug0_v5_half_first_deck_$(date +%Y%m%d_%H%M%S)"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint data/experiments/mug0_residual_v5_20261106_100328/model_000200.pt \
  --num-envs 6 --episodes 5 --position-noise 0.0005 \
  --preload-gain 0.5 \
  --samples data/experiments/mug0_v5_paired5_20261106_102246/perturbations.json \
  --compare-results data/experiments/mug0_v5_paired5_20261106_102246 \
  --output "$l20_half_first"

cat "$l20_half_first/REPORT.md"
cat "$l20_half_first/MARGIN_COMPARISON.md"
xdg-open "$l20_half_first/diagnostics.html"
```

第一批原全倍率29/30、六方向4/5。该表并非盲测，作用为检查固定倍率跨两批已有样本的取舍。
CPU计划已检查，实际交叉复测待用户执行。

## 第一批半倍率已完成，固定参数做20样本确认

第一批半倍率26/30→29/30、完整六方向2/5→4/5，修复3/损害0；两批累计53/60→58/60、六方向5/10→8/10，修复5/损害0。
第一批角度等误差仍有取舍；已有两表用于方法/倍率选择。下一步固定0.5倍率和检查点，用新seed20平移，不训练：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_confirm="data/experiments/mug0_v5_half_confirm20_$(date +%Y%m%d_%H%M%S)"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint data/experiments/mug0_residual_v5_20261106_100328/model_000200.pt \
  --num-envs 6 --episodes 20 --position-noise 0.0005 \
  --preload-gain 0.5 --seed 20261112 \
  --output "$l20_confirm"

cat "$l20_confirm/REPORT.md"
xdg-open "$l20_confirm/diagnostics.html"
```

不要带旧samples/baseline-results/compare-results：本轮是全新表，两组仍读取同一新表配对。
20个独立平移，每组120次方向试验；大约40分钟为基于此前耗时的估计，不是保证，已有任务会排队。
CPU计划已检查，实际确认评估尚未运行。先看修复/损害与完整六方向，再看误差，不能只看平均奖励或方向通过率。

## 20样本确认后的重要修正

`mug0_v5_half_confirm20_20261106_135255` 表面记录104/120→113/120、六方向11/20→13/20，但4个新增损害在2.5秒动作开始前两组状态已经不同。
全120次中16次2.5秒位置差>0.1mm（诊断阈值），整个前缀目标完全相同，最大位置差25.11mm。
因此暂停倍率搜索/训练，先实现新进程零–零过程复现及前缀审计；不能把当前修复/损害全部归因于残差动作。
详见工程复盘第33节及 [前缀审计JSON](MUG0_CONFIRM20_PREFIX_AUDIT.json)。
