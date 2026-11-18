# v4：先消除负存活收益的奖励漏洞

## 证据与问题

v3固定样本评估中，零残差30/30通过，策略仅6/30通过。旧v2策略在同样本为30/30。
零残差轨迹在两次评估中的连续误差一致，支持优先检查策略/学习目标，而非先归因于环境漂移。

v3训练仅执行87次实际更新，完整回合方向平均通过率约4.17%，存活重力样本平均奖励约-0.405。
新成本与较尖锐的位姿/速度乘法奖励组合，使部分真实成功轨迹也持续获得负分。
不能把更小的value loss解释成控制更好；错误目标也可能被拟合。

离线回放已保存的240Hz轨迹，严格通过标签与原结果一致：

|已记录控制|原通过数|v3存活均奖励为负|v4回放存活均奖励为负|v3最低折扣回报|v4回放最低折扣回报|
|---|---:|---:|---:|---:|---:|
|零残差|30/30|14/30|0/30|-22.04|0.0121|
|v3策略|6/30|16/30|0/30|-40.45|-4.59|

零残差本来能成功，一些轨迹的折扣回报却低于一次失败罚分-5，说明目标存在
“通过提前失败规避后续负代价”的具体反例。这不是证明策略有意利用漏洞，也不是退化的唯一原因。

审计输出：[reward_audit.json](../data/experiments/residual_v3_reward_audit_20261105/reward_audit.json)、
[审计报告](../data/experiments/residual_v3_reward_audit_20261105/REPORT.md)。
v4列是固定旧轨迹上的新奖励计算，**不是v4策略结果**。

## 最小修正

v3：`live_reward = positive_quality - shaping_costs`

v4：`live_reward = positive_quality / (1 + shaping_costs)`

位姿、速度、偏置等代价仍降低奖励，但有效存活阶段不再积累负分。
第一次严格失败仍-5、随后吸收为0，最终完整成功仍+5。
准备阶段仍完全沿reference，不参与学习。

本轮不同时调小探索、不改幅度/限速、不改PPO学习率、不改物理/验收条件，以便区分奖励形式的影响。
v4保留120维观测、原SDF、PD、物理mimic、质量/摩擦和13秒过程。
消除上述反例不保证训练收敛，也不保证全通过策略之间的裕量改善。

v4是新任务语义，不能静默续训v3价值网络。v1/v2/v3入口与检查点保持可用。

## 离线复核，不启动仿真

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

.venv_l20/bin/python scripts/audit_isaacsim_margin_reward.py \
  --results data/experiments/residual_v3_paired_20261105_174700 \
  --output "data/experiments/reward_audit_repeat_$(date +%Y%m%d_%H%M%S)"
```

回放只用实际保存的状态，不修改原轨迹或通过标签。数据一次解压后计算，不在每个物理步重复读取压缩文件。

## 只做200批次v4短训

```bash
l20_v4_out="data/experiments/residual_v4_mug20_$(date +%Y%m%d_%H%M%S)"

bash scripts/isaacsim_residual_v4.sh \
  --mode train --object mug --candidate 20 \
  --num-envs 24 --iterations 200 --rollout-steps 32 \
  --residual-scale 0.15 --residual-rate 0.2 --position-noise 0.0005 \
  --output "$l20_v4_out"
```

默认采样批次数也已设为200，未运行长训。
`reward/shaping_cost` 是折扣因子中的代价，`reward/costs` 为直接减去的代价（v4为0），
`reward/positive` 为实际非负存活质量。跨版本奖励数值不能用于宣称控制性能改善。

## 同样本复测，先对比v3，再对比原v2

```bash
l20_v4_eval="data/experiments/residual_v4_paired_$(date +%Y%m%d_%H%M%S)"

bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint "$l20_v4_out/model_000200.pt" \
  --num-envs 6 --episodes 5 --position-noise 0.0005 \
  --samples data/experiments/residual_v2_paired_0p5mm_20261105_153925/perturbations.json \
  --compare-results data/experiments/residual_v3_paired_20261105_174700 \
  --output "$l20_v4_eval"

cat "$l20_v4_eval/REPORT.md"
cat "$l20_v4_eval/MARGIN_COMPARISON.md"
```

与v3比较只说明能否修复退化；恢复到6/30以上不等于优于零残差或原v2。
保留比较报告后，还可用同一结果离线比较原v2，命令会重写当前目录的对照报告：

```bash
cp "$l20_v4_eval/MARGIN_COMPARISON.md" "$l20_v4_eval/VS_V3.md"
cp "$l20_v4_eval/margin_comparison.json" "$l20_v4_eval/vs_v3.json"

.venv_l20/bin/python scripts/compare_isaacsim_margin.py \
  --results "$l20_v4_eval" \
  --previous data/experiments/residual_v2_paired_0p5mm_20261105_153925
```

判断：先恢复严格通过能力，再看与零残差的滑移/转角和动作偏置是否改善。没有改善时保留零残差，不继续无条件长训。
后续是否需要更小探索或残差范围，依据这次对照再决定。

## 显示报错与检查边界

用户报告的Chromium zygote/broken pipe发生在评估Complete之后。
结果目录中的HTML及30份侧栏数据文件已存在，该错误属于浏览器显示层，不推翻已写入的评估数据。
可先阅读REPORT.md、JSON与CSV；本轮没有修改或重装浏览器。

CPU检查覆盖非负存活奖励、代价仍有作用、v3/v4相同动作下命令/观测/验收一致、失败一次罚分及吸收零奖励。
离线审计复现全部60条旧轨迹的严格标签；v4 GPU训练和正式评估尚未执行。
