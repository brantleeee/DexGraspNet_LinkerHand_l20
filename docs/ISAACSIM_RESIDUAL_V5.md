# v5：零重力稳定末段的残差预加载

2026-11-06。独立任务 `isaacsim_sdf_residual_v5`，不修改v4或跨载旧检查点。
目标是检验杯子0重力后33–46ms早期滑移是否与干预过晚/接触预加载不足有关，不承诺改善。

## 行为合同

|时间|参考与动作|训练样本|奖励/验收|
|---|---|---|---|
|0–2秒|原五次平滑闭合，策略输出忽略|不参与PPO|原穿透历史持续检查|
|2–2.5秒|零重力，原闭合参考，策略忽略|不参与PPO|与v4相同|
|2.5–3秒|零重力，允许有界限速残差，滤波记忆连续|参与PPO|没有新增正存活奖励；超穿透/新失败仍原终止惩罚|
|3–13秒|开启重力，残差及滤波记忆连续，物体不重置|有效非失败样本参与PPO|完全沿用v4严格历史验收及非负质量收益|

默认 `--preload-seconds 0.5`，必须完全位于settle内并对齐1/60秒控制周期。
120维观测沿用历史与时间，时序边界固定，不需要增添随机阶段状态。
16维残差幅度0.15rad、偏置限速0.2rad/s，后者不等于实际关节速度硬上限。
SDF256、物理mimic、质量、摩擦、PD、闭合/稳定/重力时长、成功/失败阈值均不变。
仅新增干预窗口及相应有效学习样本；preload动作通过后续重力保持回报学习，不新增接触力奖励。
预加载可能推移物体，也可能增大穿透/mimic误差；不能只以接触力大判定成功。

实现：`environment_v5.py` / `preload.py`；旧env v2/v3/v4、history/control/reward/PPO未改。
通用基线、训练与配对入口已支持v5；默认基线仍v4，不静默切换已有任务。

## 已完成验证

- CPU测试34项通过，包括v5零动作与v4逐步目标、观测、奖励、终止相同，前缀忽略非零动作，预加载可动作、无额外零重力收益，以及预加载失败惩罚/吸收与边界校验。
- 已保存v2/v3/v4训练代码散列逐项匹配；v5入口拒绝v4检查点。
- 本机六环境GPU短检查：零动作与固定小幅非零动作两段各200策略步（各800物理步），跨过重力开启点。
- 输出 `data/experiments/mug0_v5_preload_smoke_20261106`，138个SDF碰撞体、30个物理mimic；观测120维。
- 零动作目标参考误差与非零前缀误差均约1.19e-7rad；生效时间2.5秒、重力开启3秒。
- 固定raw action=0.1检查中，预加载结束六方向接触力约2.57–10.35N（零动作约0–4.13N），窗口内最大穿透约0.219mm、mimic最大约0.0167rad，物体推移最大约0.50mm。
- 这是接口/响应短检查，不是学习策略、更非完整10秒稳定性改善；没有运行长训。

原始240Hz记录为 `zero_diagnostics.npz` / `gated_nonzero_diagnostics.npz`，边界统计在 `preload_response.json`。
已测CPU目标等价不声称跨进程PhysX轨迹逐位相同。

## 1. 相同五样本的新版本零残差基线

在同一终端依次执行，换终端须重设实际目录变量。不要同时启动多个Isaac Sim进程，已有任务会经锁排队。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_v5_base="data/experiments/mug0_v5_baseline5_$(date +%Y%m%d_%H%M%S)"

bash scripts/validate_isaacsim_reference_baseline.sh \
  --task-version v5 --preload-seconds 0.5 \
  --object mug --candidate 0 --num-envs 6 --episodes 5 \
  --position-noise 0.0005 \
  --samples data/experiments/mug0_rl_baseline5_20261105_213034/perturbations.json \
  --output "$l20_v5_base"

cat "$l20_v5_base/REPORT.md"
cat "$l20_v5_base/admission.json"
```

预期零动作不因新增窗口改变参考控制，但标签必须复核，不提前假定仍26/30。
若准备阶段有效且 `suitable=true`，才进入下步；若变为全通过，先核对与v4基线差异，不强行训练。
此处采样表与原v4首次配对相同，适合版本对照，后续新seed独立保留。

## 2. 独立短训200采样批次

```bash
l20_v5_train="data/experiments/mug0_residual_v5_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/train_isaacsim_reference.py \
  --baseline-results "$l20_v5_base" \
  --num-envs 24 --iterations 200 --rollout-steps 32 \
  --output "$l20_v5_train"
```

自动根据基线选择v5及0.5秒窗口，检查散列和物理参数，不加载v4策略。
训练仍有随机探索；200采样批次不等于200次完整抓取，实际更新数与完整回合看metrics。
不同时改奖励、增益、摩擦、扰动或验收。

## 3. 同样本配对，并与v4作版本对照

```bash
l20_v5_eval="data/experiments/mug0_v5_paired5_$(date +%Y%m%d_%H%M%S)"
bash scripts/evaluate_isaacsim_residual_paired.sh \
  --checkpoint "$l20_v5_train/model_000200.pt" \
  --num-envs 6 --episodes 5 --position-noise 0.0005 \
  --baseline-results "$l20_v5_base" \
  --samples "$l20_v5_base/perturbations.json" \
  --compare-results data/experiments/mug0_residual_paired5_20261105_220153 \
  --output "$l20_v5_eval"

cat "$l20_v5_eval/REPORT.md"
cat "$l20_v5_eval/MARGIN_COMPARISON.md"
xdg-open "$l20_v5_eval/diagnostics.html"
```

先检查独立基线复现及版本对比中零残差是否一致，再解释策略改善。
v5逐试验JSON/CSV新增 `preload_response`，含开重力时接触力、窗口内穿透/mimic、关节运动、物体推移；240Hz曲线仍完整保存。
目标是修复早期gy+、保留原成功方向，提高完整六方向通过数，同时控制连续误差代价。
v4与v5该表均29/30、完整六方向4/5；v5连续误差改善，剩余组3 gy+未修复。实际分析与第二批命令见 [结果分析](MUG0_V5_RESULT_ANALYSIS.md)。
若同样本收益出现，固定新检查点再使用新seed那份既有扰动表复测；不能将同样本优化称泛化。
若仍在前33–46ms滑移，即使接触力变大，也需检查接触位置/法向、物体运动与动作空间，不直接增加训练。

## 复现短检查（可选，不训练）

```bash
bash scripts/isaacsim_residual_v5.sh \
  --mode smoke --object mug --candidate 0 \
  --num-envs 6 --steps 200 --preload-seconds 0.5 \
  --output "data/experiments/mug0_v5_smoke_$(date +%Y%m%d_%H%M%S)"
```

## 已完成同样本结果

v5正式训练与配对已完成，独立基线26/30复现、策略29/30、修复3/损害0，完整六方向4/5。相同29次通过试验的滑移/转角/mimic误差下降，但剩余gy+在预加载中丢失有效接触，仍早期失败。当前冻结检查点，先第二批同表版本对照，不扩大长训。见 [详细复盘](MUG0_V5_RESULT_ANALYSIS.md)。
