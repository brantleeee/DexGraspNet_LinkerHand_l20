# L20 静态参考残差策略

新增独立 [SDF碰撞reference入口](../docs/MJLAB_SDF.md)：本机MuJoCo/MJWarp3.16支持原生网格SDF，旧凸碰撞reference保留。

## 确定性候选细化与动力学验证

**2026-11-02 协议修正：** 独立验证入口默认使用 `--wrist-motion kinematic`。
旧 mocap 位置更新没有向接触求解器提供平移速度；移动摩擦平台的正对照显示，
平台移动 10 cm 而物体留在原地。旧实验的 `lift` 结果应保留作历史记录，不能全部
解释为抓取失效。新协议给手腕 freejoint 提供位置、速度和加速度，以 root armature
`1e6` 近似理想受控机械臂；没有对物体施加辅助力，但这不构成真实机械臂能力验证。
`--wrist-motion mocap` 只用于复现旧协议。请用新输出目录重测，不能跨协议续跑。
旧 v1–v5 强化学习任务尚未迁移本次独立验证入口的手腕修正。

细化审计也已收紧：逐接触目标点必须收敛，并在限位/mimic 投影导出后再次检查。
仅满足连杆最近距离阈值不再算作细化成功。注意两侧 geom 各 1 mm margin 在
本地引擎中合成为 2 mm 接触对 margin；当前固定物体细化参数仍保留该历史值作对照，
缩小或渐进启用 margin 属于下一轮独立消融，不能将“减少穿透”直接当作抓取成功。

新增入口独立于下文 v1–v5 策略，不读取检查点、不修改奖励、不重置拇指参考角。
默认后端直接使用 `mjlab.sim.Simulation` / MuJoCo-Warp。

### 全部候选逐一测试（默认）

不传 `--indices` 时自动选择源 `.npy` 中的全部候选，也可显式传 `--all`。
不按能量、几何质量或历史通过状态预筛选；每个原始候选都进入验证。
默认测试六方向重力及抬升，共七个 case。物体 scale 取每个候选保存值，
不会改变手腕或物体比例。可用 `--scale 0.06` 仅选择已生成的相应尺度。

先预览选择范围（不启动仿真、不创建输出目录）：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
bash scripts/mjlab_refinement.sh \
  --result-file data/experiments/l20_jenga_006_screen_20261030/results/2_of_Jenga_Classic_Game.npy \
  --all --output data/experiments/mjlab_jenga_all_manual --dry-run
```

完整逐一测试：

```bash
bash scripts/mjlab_refinement.sh \
  --result-file data/experiments/l20_jenga_006_screen_20261030/results/2_of_Jenga_Classic_Game.npy \
  --all --execution sequential \
  --cases gx+ gx- gy+ gy- gz+ gz- lift \
  --friction 3 \
  --output data/experiments/mjlab_jenga_all_manual
```

同一个候选的原始与可用细化构型跑完全部 case，才启动下一个候选。
细化映射拒绝或局部范围超限不会跳过原始候选。每个候选使用独立进程，退出后释放
GPU 模型及图缓存；运行进度打印为 `[当前序号/候选总数]`。
单个候选发生程序/仿真异常时，记录日志并继续剩余候选；这类异常不冒充物理失败。

中断后重新执行**相同命令并添加 `--resume`**，保留原输出路径。
已完整测完但物理上失败的候选也会跳过；未完成或运行异常的候选写入新的
`attempt_0002/` 等目录后重试。旧尝试不覆盖、不删除。
续跑必须保持源数据、代码、资产和测试参数一致；改变参数请使用新输出目录。
续跑现在也检查运行库版本、接触点 JSON、轨迹文件完整性及导出资产哈希。
同一输出目录有排他锁；运行中代码或资产发生变化会停止汇总，防止混用不同版本。
只有报告成功生成后，根目录的 `summary.json` 才会标记 `completed: true`。
导出 XML 保存实际使用的仿真选项；细化收敛速度检查包含被动 DIP。

可选：首次运行时加 `--compare-cpu` 可对每个候选补充同控制器 CPU 对照；
加 `--skip-refinement` 可仅测原始构型。这些参数续跑时必须保持一致。
仅需快速筛选时可明确指定 `--cases gz- lift`。

输出结构：

```text
mjlab_jenga_all_manual/
  sweep_settings.json       # 输入、参数、代码与资产哈希
  progress.json             # 每个候选的完成/异常/中断状态
  errors.json               # 运行异常，区别于抓取物理失败
  results.json              # 全部已完成候选的逐 case 指标
  summary.json              # 总数、完成数、异常数及严格通过候选
  REPORT.md                 # 汇总报告
  candidate_000/
    attempt_0001.log        # 本候选的详细运行日志
    attempt_0001/           # 独立细化报告、reference、XML 与轨迹
```

`summary.json` 的 `completed: true` 表示全部候选已测完，不表示抓取全部成功。
`fully_passed` 才表示通过全部请求 case 的候选/构型。
存在运行异常时全部候选仍会被尝试，程序最后退出码为 1；正常完成（即便抓取全失败）
退出码为 0；Ctrl+C 中断退出码为 130。

### 按 scale 批量验证（显式选择）

传 `--execution batch` 时，同一物体与 scale 的
原始、细化候选在独立 world 中批量验证，各自恢复生成的手腕、关节和控制状态。
`--backend cpu` 显式选择 C MuJoCo；GPU 出错不会自动退回 CPU。

```bash
bash scripts/mjlab_refinement.sh \
  --result-file data/experiments/l20_jenga_006_screen_20261030/results/scales/scale_0p06/2_of_Jenga_Classic_Game.npy \
  --indices 12 --execution batch --cases gz- lift --friction 3 --compare-cpu \
  --output data/experiments/mjlab_jenga_candidate12_new
```

批量验证时将 `--indices` 改为 `0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15`。
需要六方向测试时指定 `--cases gx+ gx- gy+ gy- gz+ gz- lift`。
这是六方向**重力**测试；没有额外叠加论文中的六方向外力。
`--skip-refinement` 仅验证原始候选。输出目录必须不存在。

流程及物理设置：

1. 校验网格身份、缩放、关节限位及 mimic，保留原始合法性报告。
2. 从已标注的手部接触候选点重建逐指接触对应，物体目标点固定在世界坐标。
   默认要求拇指、食指、中指；可用 `--required-fingers` 调整。
   目标间隙超过 20 mm 或缺少拇指对向表面时拒绝细化，仍保留原始验证。
3. 细化固定物体和手腕，关闭重力，采用法向接触 `condim=1`，默认 1 mm
   geom margin。用 `J^T f` 和 `T^T` 将虚拟接触力映射至 16 个独立执行器，
   DIP 保持被动；阻尼交由隐式积分处理，执行器力矩仍受限。
   MuJoCo-Warp 对 mesh MULTICCD + 非零 margin 有限制，因此仅细化模型关闭
   MULTICCD；同时将机械 equality/关节限位的 solref 收紧至 `0.002 1`。
   最大局部偏移默认 0.25 rad，最多细化 0.5 s。
4. 保存细化前的原始状态、精确 mimic/限位投影量及投影后的几何审计。
   独立关节投影 >0.001 rad 或 mimic 投影 >0.01 rad 不视为有效细化。
   未通过审计的细化姿态可作为诊断轨迹运行，但不会被最终 `passed` 接纳。
5. 验证重新编译自由物体场景，恢复原碰撞设置、摩擦与 equality 参数，
   移除细化 margin 和虚拟力。A/B 共用原 XML 固定 PD；本入口尚未加入
   自适应控制、接触力 QP 或额外 squeeze，不能视为 Dexonomy 的完整复现。

标准协议为 0.3 s 零重力夹持建立、0.2 s 重力加载。抬升测试再保持 0.5 s，
用 1 s 抬升手腕 10 cm，最后保持 2 s。六方向重力测试在加载后保持 2 s。
无地面、焊接或物体支撑。`--protocol legacy` 使用旧验证器的 0.3 s settle、
加载时立即开始抬升和默认 2 s loaded 时长，但控制器固定为 PD。
CPU/GPU 对照验证**同一份 GPU 细化姿态**，不混入独立 CPU 细化的差异。

`dynamics_pass` 要求全过程相对手掌滑移 ≤5 mm、旋转 ≤15°、穿透 ≤1 mm、
关节越限 ≤0.001 rad、mimic 误差 ≤0.01 rad。末尾连续至少 1 s 满足逐指
正法向承载、线速度 <0.02 m/s、角速度 <0.3 rad/s；抬升还需物体实际上升 ≥8 cm。
`passed` 额外要求源姿态合法和几何/接触映射合格。手指 0.2 rad/s 旧奖励阈值
单独记录为速度诊断，未作为本入口的抓取成功条件。默认沿用历史手部自碰撞 mask，
可用 `--self-collision` 显式开启；未开启的实验不能作为自碰撞合格证明。

输出包括 `results.json`、`summary.json`、`settings.json`、资产哈希，
每个候选的接触映射、几何审计、细化报告和原始/细化 reference。
`trials_000/` 中共享场景 XML；每条 NPZ 保存实际初态，以及 20 ms 间隔的
qpos/qvel/ctrl、手腕位姿、逐指法向力、滑移、旋转、穿透、耦合误差和保持时长。
`summary.json` 的 `fully_passed` 仅收录通过所有请求 case 的候选。
源候选能量仅存于细化结果的 `parent_optimization_metrics`，不冒充细化后的能量。

使用实际记录回放渲染，必须同时指定共享 XML 和对应轨迹：

```bash
MUJOCO_GL=egl .venv_l20/bin/python scripts/render_grasp_dynamics.py \
  data/experiments/mjlab_jenga_candidate12_new/trials_000/scene_lift_mjlab.xml \
  --trajectory data/experiments/mjlab_jenga_candidate12_new/trials_000/candidate_012_refined_lift_mjlab.npz \
  --output data/experiments/mjlab_jenga_candidate12_new/refined_lift.mp4
```

`prepare_mjlab_grasp.py` 现支持任意正 scale，并可用 `--scene` 指定资产场景。
导出时从同一候选重建手腕、关节和物体 scale；原 v4/v5 策略仍有独立的拇指
初态规则，应使用本确定性入口评价生成候选本身。

参考为 scale 0.06、组内候选 7（原始候选 18）。物体网格只缩放一次；保存的手掌位姿、关节角与该比例对应。16 个独立关节输出残差，5 个 DIP 通过原模型的 equality 联动。零位是策略的关节目标偏置，受力后实际关节角仍会改变。

## 版本与稳定性诊断

旧任务 v1 的 10000 轮检查点在 `data/experiments/mjlab_grasp_006_10000_20261030_031408/model_9999.pt`。日志最后的严格成功奖励仍为 0。动力学诊断发现抬升约 10 cm 的同时，杯子旋转约 40–48°，且高比例策略分量超过硬限幅边界。保持位置与抬升奖励没有连续约束旋转和速度，因此总奖励高不代表稳定。

v2 保留原物体质量、比例、摩擦、关节力矩限制和静态参考，修改控制和目标：

- 目标残差 `0.2 * tanh(action)` rad，继续按 URDF / mimic 可行范围裁剪。
- 目标变化速度限制为 1 rad/s（50 Hz 下每步最多 0.02 rad）。回合重置清除滤波状态；零动作从静态零位开始，运动中回到零位也经过限速。
- 观测加入实际滤波目标残差，避免隐藏控制器状态。
- 位置和旋转共同决定保持、抬升奖励，增加相对预定抬升轨迹的速度惩罚、角速度惩罚和原始动作幅值惩罚。
- 回合从 2.5 s 延长至 5 s，抬升仍为 0.5 s 后用 1 s 抬高 10 cm，其余时间考查保持。

内环仍为 XML 的固定 PD 控制器，尚未迁移原 NumPy 自适应柔顺控制器。杯子自由运动，没有焊接或隐藏支撑。候选 7 曾未通过动力学验证，不能将其视作专家成功示范。v2 是未通过稳定性验证的实验任务，不能作为已修复版本：200 轮试训后的 320 个重复初态评估回合成功数为 0；1 rad/s 的目标限速也会导致闭合过慢。v1 与 v2 保留用于复现实验；当前新运行默认使用下文的 v4，v2 必须显式选择。

## 运行

使用独立 mjlab 环境，避免与 `.venv_l20` 的 Torch / MuJoCo 版本混用。包装脚本默认解释器为 `/home/lee/robot_learning/mjlab/.venv/bin/python`，可用 `MJLAB_PYTHON` 环境变量指定。

```bash
# 新任务的 GPU 冒烟验证
bash scripts/mjlab_grasp.sh --mode smoke --task-version v2 --num-envs 4 --steps 300 \
  --output "data/experiments/mjlab_v2_smoke_$(date +%Y%m%d_%H%M%S)"

# 实验短训；先通过评估再决定是否长训。旧 v1 模型不能直接续训到 v2
bash scripts/mjlab_grasp.sh --mode train --task-version v2 --num-envs 64 --iterations 200 \
  --output "data/experiments/mjlab_v2_train_$(date +%Y%m%d_%H%M%S)"

# 将 CHECKPOINT 替换为训练输出下的 model_*.pt
bash scripts/mjlab_grasp.sh --mode evaluate --checkpoint CHECKPOINT --num-envs 16 --steps 1000 \
  --output "data/experiments/mjlab_eval_$(date +%Y%m%d_%H%M%S)"

MUJOCO_GL=glfw bash scripts/mjlab_grasp.sh --mode view --checkpoint CHECKPOINT \
  --output "data/experiments/mjlab_view_$(date +%Y%m%d_%H%M%S)"
```

评估保存 `evaluation.json` 与 `trajectory_env0.csv`，包含完整回合成功数、连续稳定保持时间、旋转、位置误差、速度与动作幅值。v2 的成功要求抬升后高度 > 8 cm、位置误差 < 5 mm、旋转 < 15°、线速度 < 0.02 m/s、角速度 < 0.3 rad/s，连续满足至少 1 s。v1 的保持窗口为 0.5 s，以适配旧 2.5 s 回合。当前评估要求保持窗口一直延续到超时前，掉落回合直接失败；固定初态的重复回合不构成扰动鲁棒性验证。

旧检查点没有任务版本文件时自动采用 v1；新运行保存 `task_snapshot.json`。显式请求不兼容任务版本时拒绝加载。

CPU 控制器与奖励检查：

```bash
UV_CACHE_DIR=/tmp/l20-mjlab-uv uv run --no-project \
  /home/lee/robot_learning/mjlab/.venv/bin/python \
  -m unittest discover -s mjlab_grasp/tests -v
```

查看训练曲线：

```bash
bash scripts/tensorboard.sh
```

浏览器打开 `http://127.0.0.1:6006`，选择对应的实验目录。启动脚本使用项目的 `.venv_l20`，其中已有 TensorBoard 所需的 `pkg_resources`；mjlab 的 Python 环境仍用于训练和评估。

## v3：按稳定保持重新设计奖励

本版保留 v1 的碰撞模型、全重力、固定 PD、16 个关节目标残差、机械限位与静态零位。没有采用 v2 的慢闭合限速。回合为 5 s，以检验抬升后保持。命令行当前新运行默认使用下文 v4；加载已有检查点则自动使用保存的任务版本。

每步奖励为下表各项乘以 `step_dt`，掉落项单独处理为一次性扣分：

| 项目 | 权重 | 条件及目的 |
|---|---:|---|
| 接触留住杯子 | +2 | 有手物接触，并按相对参考位置误差平滑衰减 |
| 保持朝向 | +2 | 接触和位置门控的连续朝向奖励，避免 v2 全部乘积接近零 |
| 抬升进度 | +3 | 有接触、留在手边、朝向良好才获得较高抬升分 |
| 稳定保持 | +8 | 抬升后位置、朝向、线速度和角速度共同决定，每步累积 |
| 严格稳定保持 | +6 | 有接触，且高度 >8 cm、位置误差 <5 mm、旋转 <15°、线速度 <0.02 m/s、角速度 <0.3 rad/s |
| 滑移与转动 | −0.75 | 线速度减去预定抬升轨迹速度，避免惩罚正常上升；有界惩罚 |
| 位置误差 | −0.5 | 有界惩罚 |
| 朝向误差 | −0.5 | 有界惩罚 |
| 原始动作超限 | −0.05 | 原始 Gaussian 动作超过 ±1 的幅度，保持非零的渐进约束 |
| 动作变化 | −0.05 | 有界惩罚 |
| 关节超速 | −0.5 | 任一手指关节（含被动 DIP）超过 0.2 rad/s 时惩罚，按最大超速比例计算 |
| 掉落 | **每次 −25** | 排除普通超时，除以 dt 抵消 RewardManager 的时间缩放 |

六项连续负奖励的绝对上限合计为 2.35/s，整个 5 s 回合最多 11.75 分；一次掉落扣 25 分，避免仅为逃避剩余负奖励而提前结束。稳定奖励按时间累积，独立评估还要求直到回合结束前连续稳定至少 1 s；中途曾经稳定但最后掉落不计成功。评估使用真实终止状态计算指标，然后手动重置，不混入自动重置后的初态。

v3 的环境接收原始动作以计算超限惩罚，但实际关节目标继续按 v1 裁剪至 ±0.2 rad 和 URDF / mimic 可行范围。`last_action` 观测维持裁剪后的 ±1，以兼容原 actor 的 77 维观测。

从原 10000 轮模型复用 actor（含观测归一化），重置 critic、优化器和训练轮数：

```bash
bash scripts/mjlab_grasp.sh --mode train --task-version v3 \
  --warm-start \
  --checkpoint data/experiments/mjlab_grasp_006_10000_20261030_031408/model_9999.pt \
  --num-envs 64 --iterations 10000 \
  --output "data/experiments/mjlab_v3_train_$(date +%Y%m%d_%H%M%S)"
```

同一 v3 检查点继续训练时，去掉 `--warm-start`，改为该 v3 检查点路径。`--iterations` 为本次额外训练轮数。v2 与 v3 观测不同，不允许直接互相加载。也可不传 `--checkpoint` 与 `--warm-start`，从头训练 v3。

每次 v3 运行保存 `rewards_used.py` 和 `reward_weights.json`，使奖励代码和权重可追溯。暖启动额外保存 `model_initial.pt`，可与训练结束模型在相同 v3 环境内对照。奖励设计和短训通过不等于已获得稳定策略，需要查看独立评估结果。

当前 v3 的 200 轮暖启动试训已经完成：GPU 与 10 项 CPU 检查通过。相同 v3 环境的训练前/后独立评估各完成 64 回合，严格成功均为 0，掉落从 0 次增至 5 次；环境 0 的抬升后角速度均值下降，但偏移增加。此结果不能证明已获得稳定抓取，不宜只根据总奖励决定继续长训。详见 [试训报告](../data/experiments/mjlab_grasp_v3_warm_pilot_20261030/REPORT.md)。

### 手指关节速度阈值

当前 URDF 的 21 个手指关节速度上限均为 1 rad/s。v3 的奖励阈值已改为其 0.2 倍，即 **0.2 rad/s**，适用于主动关节和被动 DIP，并同时检查正反向速度。

`joint_motion` 在阈值内为零；超速时按各关节 `abs(qdot)/0.2 - 1` 的最大值计算有界惩罚，权重从 −0.05 增强为 −0.5，避免单个关节超速被其余关节平均稀释。任一关节超速时，稳定保持及严格保持奖励均为零；v3 独立评估也要求连续保持窗口内手指速度达标。评估轨迹保存 `max_finger_speed_rad_s`，报告保存 `over_speed_env_steps`。

这是奖励软约束，并非物理引擎的硬速度限幅；碰撞受力时仍可能超过阈值。已有模型不会自动获得新行为，需要用新奖励重新训练。旧试训报告及保存的 `rewards_used.py` 记录的是当时的奖励，不代表当前速度阈值已通过策略训练验证。

## v4：解除拇指对静态抓取角的依赖

新运行默认 v4。拇指四个主动关节（CMC yaw/roll/pitch、MCP）在各自可行角度范围内生成绝对目标：

`q_target = (lower + upper)/2 + (upper - lower)/2 * clip(action, -1, 1)`。

范围取自 `reference.json` 的 `drive_limits`，包含 DIP mimic 联动收紧后的 MCP 限位。拇指零动作与初始复位角均取该范围中点；被动 DIP 复位按导出 XML 的 equality 计算。保存的静态拇指关节角不参与目标或复位。其余手指、手掌位置、物体 scale 0.06 与候选 7 对应关系保持原设定，其他手指仍使用静态角 ±0.2 rad 残差。

v4 继续使用当前稳定性奖励及全部 21 关节的 **0.2 rad/s 速度软约束**。本改动扩展了拇指探索范围，不构成稳定抓取成功保证。动作含义及初始拇指姿态发生变化，因此从头训练；旧版本检查点不会被当作可直接续训的 v4 模型。

从项目根目录训练 4000 轮：

```bash
bash scripts/mjlab_grasp.sh \
  --mode train --task-version v4 \
  --num-envs 64 --iterations 4000 \
  --output "data/experiments/mjlab_grasp_v4_4000_$(date +%Y%m%d_%H%M%S)"
```

不传 `--checkpoint` 或 `--warm-start`。训练结束后，使用输出目录中的 `model_3999.pt` 进行 `--mode view` 或 `--mode evaluate`；脚本自动读取版本 v4。

### 接触匹配容量与重复警告

任务已将 `SimulationCfg.contact_sensor_maxmatch` 从默认 64 调至 1024，避免多块凸碰撞网格产生的 `contact match overflow: please increase Option.contact_sensor_maxmatch beyond 64` 重复警告。训练统计表与其他正常输出保留。该配置在创建环境时生效，需要重新启动训练/查看进程；已运行的进程不会自动采用新配置。

## v5：抑制掉落时的手指目标振荡

v4 的 `model_3999.pt` 在 16 环境、500 步确定性评估中完成 32 回合，严格稳定成功 0；
8000 个环境步中有 5440 步至少一个手指关节超过 0.2 rad/s。
环境 0 的最大手指速度中位数 0.35 rad/s、峰值 8.65 rad/s，
原始动作绝对值中位数约 3.03，很多动作被裁剪到 ±1。
本次评估未掉落，但这不能排除交互视角或其他初态下的掉落。
见 `data/experiments/mjlab_grasp_v4_4000_20261030_142338/evaluation_baseline_500_gpu/`。

v5 的物理关节目标先经 80 ms 时间常数平滑，再逐环境限速：
抬升完成前（0–1.5 s）最多 **0.8 rad/s**，之后最多 **0.2 rad/s**；
50 Hz 策略步长对应最多 0.016/0.004 rad。限速状态随环境单独复位，
策略观测增加已施加目标的归一化值，避免控制器状态隐藏。
这限制的是发送给固定 PD 的目标变化速度；碰撞冲击仍可能让真实关节速度超过 0.2 rad/s。
原始 Gaussian 动作仍用于动作幅值、变化惩罚，物理目标继续满足 URDF/mimic 角度范围。

奖励降低了仅靠短暂接触、朝向和抬升获得高总分的权重，增加抬升后的连续收敛奖励，
并提高动作超限、变化、关节超速惩罚；增加抬升后目标持续运动惩罚。
严格稳定保持条件不变。动作和观测接口已变化，**v4 模型不能作为 v5 检查点续训**；
已有 v4 模型仍可自动按 v4 评估和查看。

先做短训并独立评估；只有 `successful_episodes`、`longest_stable_hold_s`
及 `over_speed_env_steps` 改善时，再考虑 4000 轮长训：

```bash
bash scripts/mjlab_grasp.sh --mode train --task-version v5 \
  --num-envs 64 --iterations 200 \
  --output "data/experiments/mjlab_grasp_v5_pilot_$(date +%Y%m%d_%H%M%S)"

bash scripts/mjlab_grasp.sh --mode evaluate \
  --checkpoint <上一步目录>/model_199.pt --num-envs 16 --steps 500 \
  --output "data/experiments/mjlab_grasp_v5_eval_$(date +%Y%m%d_%H%M%S)"
```

已完成的 100 轮 v5 短训在 16 环境、500 步独立评估中完成 39 回合，
**28 回合掉落、0 回合严格成功**；目标限速确实生效，但未解决抓取。
因此 v5 保留为可复现实验，当前不建议直接跑 4000 轮。
在已训练 v4 模型上仅于抬升后施加 0.2 rad/s 目标限速，
严格成功仍为 0，超速环境步数从 5440 增至 5593；
把位置控制阻尼从 0.05 提至 0.08 或 0.2 Nm·s/rad 也未成功，
其中 0.2 导致 36 次掉落和明显关节失稳。对应评估文件保存在
`data/experiments/mjlab_grasp_v4_4000_20261030_142338/evaluation_*_500*/`。

可重现现成模型的抬升后限速对照，旧检查点会自动载入 v4：

```bash
bash scripts/mjlab_grasp.sh --mode evaluate \
  --checkpoint data/experiments/mjlab_grasp_v4_4000_20261030_142338/model_3999.pt \
  --hold-target-rate 0.2 --num-envs 16 --steps 500 \
  --output "data/experiments/mjlab_v4_hold_filter_$(date +%Y%m%d_%H%M%S)"
```

下一步应先生成更多杯子 scale 0.06 的候选，并逐个通过 MuJoCo 六方向重力、抬升、
接触穿透筛选，再用物理通过的静态抓取作训练参考。
当前 16 个杯子候选中满足几何阈值的 0、7、12、15 在现有 MuJoCo
`gz-` 与抬升测试中全部失败；仅通过几何优化不足以确认稳定。

下面命令生成 64 个候选并先做重力向下与抬升筛选；`scale_validation.json`
还需检查几何阈值，`results.json` 还需检查 `peak_object_contact_depth_mm <= 1`。
此步骤可能耗时较长，只有通过筛选的候选才值得进一步测试六方向重力并训练。

```bash
run_name="l20_cup_006_screen_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/generate_small_grasps.py \
  --object-code core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03 \
  --scales 0.06 --batch-size-per-scale 64 --n-iter 6000 \
  --hull-inflation scale-aware --name "$run_name"

OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/validate_grasp_dynamics.py \
  --result-file "data/experiments/$run_name/results/scales/scale_0p06/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03.npy" \
  --indices $(seq 0 63) --scale 0.06 --friction 3 --controller adaptive \
  --cases gz- lift --seconds 2 \
  --output "data/experiments/$run_name/dynamics_screen"
```

如果候选 `N` 通过，再用同一 `--result-file`、`--indices N` 测
`--cases gx+ gx- gy+ gy- gz+ gz- lift`。通过后可导出新的训练参考：

```bash
.venv_l20/bin/python scripts/prepare_mjlab_grasp.py \
  --result-file "data/experiments/$run_name/results/scales/scale_0p06/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03.npy" \
  --dynamics-dir "data/experiments/$run_name/dynamics_screen" \
  --candidate-index N \
  --output "data/mjlab_grasp/${run_name}_candidate_N"
```

新参考需在训练与评估时都用相同的 `--reference` 路径。
MuJoCo 自适应验证器与 mjlab 固定 PD 仍不同；导出后应先用零动作
`--mode smoke` 和短训独立评估确认新环境。
