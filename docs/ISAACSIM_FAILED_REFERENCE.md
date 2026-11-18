# 独立失败reference：先复核，再训练，再配对

杯子20原型已阶段收尾。本流程面向原本会失败的reference，默认积木 `jenga` 原始候选7。
不加载杯子检查点，不改变v4观测/奖励、物理、控制或严格通过标准。

## 1. 无检查点的零残差复核

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

l20_base="data/experiments/jenga7_rl_baseline_$(date +%Y%m%d_%H%M%S)"

bash scripts/validate_isaacsim_reference_baseline.sh \
  --object jenga --candidate 7 \
  --num-envs 6 --episodes 1 --position-noise 0.0005 --seed 20261107 \
  --output "$l20_base"

cat "$l20_base/REPORT.md"
cat "$l20_base/admission.json"
```

没有策略检查点或随机动作；只加载同一生成候选的v4 RL场景。
记录每个物理步首次严格失败的时间、阶段、原因，保留240Hz诊断。
`perturbations.json` 保存一个平移样本，在六方向复用；后续评估必须读取同一表。
一个样本足以做流程准入检查，不证明全部扰动下准备阶段都有效。
需要扩大基线覆盖时可改 `--episodes 5`，后续评估必须匹配相同episodes和num-envs。

训练准入：准备阶段没有失败、已获取接触、至少一个方向在重力阶段失败。
闭合穿透、获取失败会阻止当前重力阶段学习任务，需先改善reference或另设闭合阶段任务。
全部通过则没有本轮要修复的失败，也不通过该失败修复入口；原训练入口仍可用于其他研究目的。

## 2. 已完成的本机积木7基线

已运行：`data/experiments/jenga7_rl_baseline_20261105/`，每轴±0.5mm、seed20261107、一个平移的六方向。

|方向|通过|首次失败全局时间|距开启重力时间|原因|
|---|---|---:|---:|---|
|gx+|否|3.0375s|37.5ms|滑移|
|gx-|否|3.0417s|41.7ms|滑移|
|gy+|否|3.0583s|58.3ms|滑移|
|gy-|是|—|—|—|
|gz+|是|—|—|—|
|gz-|是|—|—|—|

准备阶段均合格，3个失败发生在重力阶段，满足阶段准入。
**但阶段正确不等于可控性已经证明**：策略60Hz、目标偏置限速0.2rad/s，最早37.5ms内的偏置变化上界约0.0075rad。
当前动作能否及时建立足够接触力，必须由训练及实际动作记录验证。若不能修复，优先检查响应时间、接触与动作可达性，不盲目延长训练。

## 3. 独立策略短训

可直接复用已经完成的本机基线：

```bash
l20_base="data/experiments/jenga7_rl_baseline_20261105"
l20_jenga_out="data/experiments/jenga7_residual_v4_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/train_isaacsim_reference.py \
  --baseline-results "$l20_base" \
  --num-envs 24 --iterations 200 --rollout-steps 32 \
  --output "$l20_jenga_out"
```

入口从基线自动确定物体、原始下标、reference目录、扰动大小、控制和阶段参数，核对实际v4计划与基线一致，
仅改变训练并行环境数。没有 `--checkpoint` 参数，actor/critic/优化器从新策略开始。
可加 `--plan-only` 检查实际训练命令而不创建目录/运行GPU。
不合适的基线、原始输入改变或代码散列改变都会拒绝启动。
完成后保存 `baseline_admission.json`，记录基线来源和本次独立训练命令。

## 4. 同样本配对评估与基线复现检查

```bash
l20_jenga_eval="data/experiments/jenga7_residual_paired_$(date +%Y%m%d_%H%M%S)"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint "$l20_jenga_out/model_000200.pt" \
  --num-envs 6 --episodes 1 --position-noise 0.0005 \
  --baseline-results "$l20_base" \
  --samples "$l20_base/perturbations.json" \
  --output "$l20_jenga_eval"

cat "$l20_jenga_eval/REPORT.md"
xdg-open "$l20_jenga_eval/diagnostics.html"
```

评估前核对基线的reference、模型/控制设置及同一扰动表；新的零残差初态还需与基线保存的显式初态完全匹配。
`baseline_recheck.pass_flags_reproduced` 判断原先3/6标签是否复现。若不复现，先排查环境，而非直接计算RL增益。
评估同样记录首次失败时间/原因，并保留动作、残差、关节位置速度及力矩诊断。

本轮判据：原通过方向保持、至少修复一个原失败方向、机械约束没有明显恶化。
通过率改善后再扩大扰动样本/seed；一个平移样本的成功不能宣称泛化。
如果仍3/6或更差，检查在约40ms失效窗口内实际目标与接触力变化，决定是否应先改善静态reference或重新定义接触建立任务。

## 检查范围

CPU测试覆盖首次失败定位、准备阶段拒绝、重力失败准入及无失败不进入本修复入口；
训练计划检查证明自动选择积木7、不会载入杯子检查点、不会静默改变基线参数。
本机实际零残差6方向基线已完成；独立200批次训练和策略配对评估尚未启动。
旧杯子v2/v3/v4检查点兼容性保持。
