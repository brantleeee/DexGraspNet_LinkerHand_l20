# DexGraspNet · LinkerHand L20 抓取生成与动力学验证

本项目基于 DexGraspNet，将抓取生成适配到 LinkerHand L20 右手，并提供 MuJoCo、Isaac Gym、Isaac Sim 等验证入口。
当前主要流程为：**按物体比例生成静态抓取 → Isaac Sim 中零重力闭合抓取 → 开启重力验证 → 统计与轨迹回放**。
L20 当前 SDF 验证直接使用 Isaac Sim / PhysX，不依赖 mjlab 或训练策略。

本文面向已配置好的本机环境，更新于 **2026-11-05**。

本工程启用了项目级 [强化学习工程导师 skill](skills/rl-engineering-mentor/SKILL.md)，
后续相关任务会结合当前日志与代码说明证据、原因假设和最小验证实验，帮助理解如何定位训练退化等问题。
学习路线及练习见 [工程能力地图](skills/rl-engineering-mentor/references/learning-map.md)；
默认引用规则保存在 [AGENTS.md](AGENTS.md)，不修改用户全局配置。

完整的问题演进与实验复盘见 [现象 → 问题分析 → 解决方法 → 新现象](docs/GRASP_RL_ENGINEERING_LOG.md)。
相对原方法的工程扩展、研究改进及证据边界见 [DexGraspNet研究扩展说明](docs/DEXGRASPNET_RESEARCH_EXTENSIONS.md)。
阶段汇报与抓取对照视频见 [2026-11-08汇报材料](docs/GRASP_PROGRESS_REPORT_20261108.md)及[视频展示页](data/reports/grasp_progress_20261108/index.html)。
从验证起点到当前结论的完整研究主线见 [Isaac Sim验证至RRL研究进展](docs/ISAACSIM_TO_RRL_RESEARCH_PROGRESS.md)。
新增资源：[`assets/power-drill-obj.zip` 的L20抓取生成命令与尺寸说明](docs/POWER_DRILL_L20.md)。

最新电钻生成：[v6接触点与法向实际接入、后台生成及无WebGL前16可视化](docs/L20_V6_DRILL_GENERATION.md)。

本批中指根部弯曲异常：[实际接触、分项梯度与原始网格间隙诊断](docs/L20_V6_MIDDLE_BENDING_DIAGNOSIS.md)。

MuJoCo全手复核：[22个link、231组配对的完整弯曲扫描与原生回放视频](docs/L20_MUJOCO_FULL_HAND_MIDDLE_SWEEP.md)。
后续相关结果和实质改动会在该文档持续追加，区分已验证结论与待验证假设，不后台监控训练。

旧版已归档为[历史 README](docs/README_LEGACY_20261105.md)，保留环境编译、ShadowHand、MuJoCo 与早期实验记录。
归档中的相对路径及旧命令按项目根目录理解，历史结果不代表当前协议。

## 1. 查看已完成的五物体结果

统一入口：[index.html](data/experiments/l20_five_objects_acquire_10s_20261105/index.html)。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/experiments/l20_five_objects_acquire_10s_20261105/index.html
```

共 **80 个候选、480 个方向场景**，每个候选须六方向全部通过才计为通过。

| 物体 | 别名 | 完成候选 | 六方向全部通过 | 通过率 | 通过候选下标 |
|---|---|---:|---:|---:|---|
| 杯子 | `mug` | 16 | 10 | 62.5% | 4、5、6、13、18、20、23、27、28、30 |
| 积木 | `jenga` | 16 | 0 | 0% | — |
| 香蕉 | `banana` | 16 | 2 | 12.5% | 6、13 |
| 瓶子 | `bottle` | 16 | 2 | 12.5% | 2、12 |
| 相机 | `camera` | 16 | 4 | 25% | 10、11、12、15 |
| **合计** | | **80** | **18** | **22.5%** | |

以上均为 scale `0.06`，零重力闭合 2 秒、保持 1 秒、开启重力验证 10 秒。
杯子/积木与新增三物体来自不同生成批次；本表是已有候选汇总，不是统一生成条件下的物体难度对照。
相机候选 3 曾出现数值发散，最终改为每方向独立进程完成验证，未通过抓取标准；报告保留此区别。

统一目录通过本地链接引用以下原实验，**请保留两个源目录**：

- [杯子与积木](data/experiments/l20_all_acquire_10s_20261104_173323/REPORT.md)
- [香蕉、瓶子与相机](data/experiments/l20_extra3_20261104_225733/REPORT.md)

HTML 支持旋转、缩放、时间轴、六方向切换及手/物体分别显示，回放保存的轨迹，不重新仿真。

已完成同一批 80 个位姿的 Gym / Isaac Sim 逐方向对照：
[打开比较页面](data/experiments/l20_gym_vs_isaacsim_20261105/index.html)。
页面分别显示 Gym 原始位姿、Gym 接触调整及 Sim 的六方向结果，支持按物体筛选和只看不一致候选。
Gym 原始组最终通过 8/80，调整组通过 9/80，Sim 通过 18/80；原始组与 Sim 同时通过 5 个，调整组与 Sim 同时通过 4 个。
三组验收条件与执行过程不同，不能将该统计当作仅改变仿真引擎的对照实验。

重新生成对照（使用新的输出目录）：

```bash
.venv_l20/bin/python scripts/compare_l20_gym_isaacsim.py \
  --gym-results data/experiments/l20_gym_official100_20261105_114721 \
  --sim-results data/experiments/l20_five_objects_acquire_10s_20261105 \
  --output "data/experiments/l20_gym_sim_comparison_$(date +%Y%m%d_%H%M%S)"
```

## 2. 本机环境

| 用途 | 环境或路径 |
|---|---|
| 项目根目录 | `/home/lee/DexGraspNet_LinkerHand_l20` |
| 生成、准备输入、批处理、统计 | `.venv_l20/bin/python`（Python 3.11） |
| Isaac Sim | `/home/lee/isaac-sim`（5.1） |
| Isaac Sim Python | `/home/lee/isaac-sim/python.sh`，由启动脚本调用 |
| L20 URDF | `grasp_generation/mjcf/linkerhand_l20_right.urdf` |
| 物体别名与资源目录 | `scripts/l20_object_catalog.py` |
| 物体网格 | `data/meshdata/<object-code>/coacd/decomposed.obj` |
| 实验输出 | `data/experiments/` |

`.venv_l20` 复用本机 UniVTAC 的 Python / PyTorch 环境。原 `environment.yml` 是历史环境，
不应直接替代当前 RTX 5070 系列使用的环境。迁移电脑需另行配置 CUDA、PyTorch、PyTorch3D、TorchSDF 和 Isaac Sim；
历史编译步骤见归档 README。

不启动仿真的基本检查：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
.venv_l20/bin/python --version
test -x /home/lee/isaac-sim/python.sh
.venv_l20/bin/python scripts/run_l20_object_suite.py --help
```

Isaac Sim 安装位置不同时，在运行启动脚本前设置：

```bash
export L20_ISAACSIM_PATH=/你的路径/isaac-sim
```

## 3. 从头生成并验证五个物体

以下命令会启动长时间计算：每物体生成 16 个候选、优化 6000 次，再验证每个候选的六个方向。
`--n-iter` 是静态抓取优化次数，不是强化学习训练轮数。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_out="data/experiments/l20_five_objects_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/run_l20_object_suite.py \
  --objects mug jenga banana bottle camera \
  --scale 0.06 --batch-size-per-scale 16 --n-iter 6000 \
  --close-seconds 2 --settle-seconds 1 --open-fraction 0.6 \
  --gravity-seconds 10 --sdf-resolution 256 \
  --isolate-cases \
  --output "$l20_out"

# 运行完成后查看
xdg-open "$l20_out/index.html"
```

`--isolate-cases` 对每个候选的每个方向启动全新 Isaac Sim 进程，排除跨方向复用求解器状态的影响，
会增加启动开销。省略后，各候选仍为独立进程，但同一候选的六方向共用场景。
启动脚本通过进程锁串行使用 Isaac Sim，避免本项目多个实例同时编译 RTX 管线。

只检查计划，不创建输出目录、不启动生成或仿真：

```bash
.venv_l20/bin/python scripts/run_l20_object_suite.py \
  --objects mug jenga banana bottle camera \
  --scale 0.06 --batch-size-per-scale 16 --n-iter 6000 \
  --gravity-seconds 10 --isolate-cases \
  --output "$l20_out" --plan-only
```

输出路径必须是新目录。生成位姿另存于 `data/experiments/l20_<实验标识>_<物体>_<散列>/`，
具体路径和实际子命令记录在套件目录的 `plan.json`。

### 物体与比例

| 别名 | object-code |
|---|---|
| `mug` | `core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03` |
| `jenga` | `2_of_Jenga_Classic_Game` |
| `banana` | `ddg-gd_banana_poisson_002` |
| `bottle` | `sem-Bottle-437678d4bc6be981c8724d5673a063a6` |
| `camera` | `sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5` |

`scale` 是归一化物体网格顶点的线性乘数，手部不缩放。每个比例重新初始化、重新优化抓取。
保存的 `qpos` 平移单位为米，已经对应候选比例，**不能再乘一次 scale**；角度为弧度。
不同文件中的候选下标可能不同，`--indices` 指输入 NPY 的数组下标。

生成助手默认使用 `scale-aware` 初始化，根据缩放后的物体尺寸选择凸包扩张距离，
这是针对小物体的实验改动；原方法的 `paper` 初始化仍可用于对照。
静态几何阈值通过与动力学稳定通过是两项独立结论。

## 4. 单独生成与验证

以瓶子为例，生成一个明确比例的新位姿文件，再验证该比例下的全部候选：

```bash
l20_gen="l20_bottle_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/generate_small_grasps.py \
  --object-code sem-Bottle-437678d4bc6be981c8724d5673a063a6 \
  --scales 0.06 --batch-size-per-scale 16 --n-iter 6000 \
  --hull-inflation scale-aware --name "$l20_gen"

l20_result="data/experiments/$l20_gen/results/sem-Bottle-437678d4bc6be981c8724d5673a063a6.npy"
l20_batch="data/experiments/l20_bottle_validation_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/batch_l20_isaacsim.py \
  --object bottle --scale 0.06 --result-file "$l20_result" \
  --protocol acquire --close-seconds 2 --settle-seconds 1 --open-fraction 0.6 \
  --steps 2400 --sdf-resolution 256 --isolate-cases \
  --output "$l20_batch/bottle"

.venv_l20/bin/python scripts/summarize_l20_isaacsim.py --results "$l20_batch"
xdg-open "$l20_batch/index.html"
```

省略 `--indices` 会验证输入文件中该比例的全部候选，不按几何能量预筛选。
香蕉、瓶子、相机必须明确传入新生成的 `--result-file`，旧资源中的位姿不满足当前 URDF 约束，会被拒绝。
杯子与积木虽有默认历史输入，使用新结果时同样建议显式传入文件。

## 5. 验证过程、轨迹与 mimic

当前使用 `--protocol acquire`：

1. 固定手腕于保存的抓取位置，按 `open-fraction` 将部分屈曲关节朝中性位置张开，保留侧摆和对掌姿态。
2. 关闭重力，用五次平滑插值在 2 秒内从张开目标闭合到保存的抓取目标。
3. 零重力保持 1 秒，再开启指定方向重力，不重置物体位姿或速度。
4. 六个轴向重力方向各测试 10 秒，每方向重新执行完整抓取过程。

L20 有 21 个转动关节，其中 16 个独立位置驱动、5 个 mimic 从动关节。
张开和闭合目标满足 URDF 的 `q_从动 = multiplier × q_主动 + offset`；
PhysX 中也建立 mimic 约束，从动关节不另设独立 PD 驱动。

这包含考虑 mimic 的闭合插值，**尚未实现避碰、接触力反馈或滑移反馈的在线轨迹规划**。
实际关节运动由接触动力学与约束求解决定。Isaac Sim 当前使用固定 PD，
未接入 MuJoCo 自适应柔顺控制或 mjlab 策略。

| 参数 | 当前设置 |
|---|---|
| 物理步长 | `1/240 s` |
| 重力阶段 | `2400` 步，即 `10 s` |
| 每方向总过程 | 闭合 480 + 保持 240 + 重力 2400 = 3120 步，即 `13 s` |
| 重力加速度 | `9.81 m/s²`，六个方向为 `gx+ gx- gy+ gy- gz+ gz-` |
| PD 增益 | `kp=5 Nm/rad`，`kd=0.05 Nm·s/rad` |
| 摩擦系数 | `3` |
| SDF 分辨率 | `256` |
| 关节约束 | URDF 限位、力矩上限及 mimic；原生关节速度上限未启用 |
| 手腕与物体 | 手腕固定；物体自由，无地面、焊接或外部支撑 |
| 手部自碰撞 | 关闭 |

原生 PhysX 关节速度上限在本机 articulation + mimic 配置中曾导致 NaN，因此未启用；
平滑目标不等于实际速度的硬约束。旧 `--protocol direct` 直接从闭合位姿施加重力，不能混入 acquire 的统计。

### 碰撞模型与通过标准

- **验证手部**：URDF 引用的 22 个完整 STL，逐 link 建立 SDF。
- **验证物体**：现有 `coacd/decomposed.obj` 整体建立 SDF。未重新做凸包或凸分解，但它本身是恢复的旧 CoACD 网格，无法恢复原始细节。
- **运行时审计**：23 个 triangle-mesh shape、0 个 convex shape。
- **抓取生成**：仍使用现有优化器的手部近似几何及物体表示，不能宣称与验证的完整 STL/SDF 表示完全一致。

每方向要求：开启重力前存在手物接触；重力阶段相对开启前位姿的最大位移 ≤5 mm、最大转角 ≤15°；
包括闭合在内的采样步接触穿透 ≤1 mm；最后连续 1 秒接触力 >0.001 N；数值有限。
六方向全部满足才标记候选通过。

穿透来自 PhysX 报告的接触点，不是网格全局精确最大穿透，也不含步进前的完整静态穿透审计。
质量按网格体积和准备阶段默认密度 `500 kg/m³` 估算，记录在 `inputs/manifest.json`，并非实物称重值。
这些结果是在指定仿真条件下的筛选，不等同于实机稳定性或官方 Gym 验收结果。

## 6. 失败诊断与续跑

普通抓取未通过会保存结果；`Nonfinite physics state` 属于数值异常，保留为未完成。
套件详细日志位于 `logs/<物体>_validate.log` 或 `logs/<物体>_resume_validate.log`，
候选异常见其 `error.json`、`numerical_failure.json`。

对已生成全部位姿、但部分验证失败的套件续跑：

```bash
.venv_l20/bin/python scripts/run_l20_object_suite.py \
  --output data/experiments/l20_extra3_20261104_225733 \
  --resume --isolate-cases
```

续跑使用原 `plan.json` 的参数，核对源位姿、URDF、网格及完成候选的物理审计，复用完成结果，
只重试未完成候选。失败尝试移入物体目录的 `failed_attempts/<时间戳>/`，不会作为通过结果复用。
改变输入或验证参数时使用新目录。`--resume` 不重新生成缺失的位姿文件；生成阶段失败需先检查日志。

独立进程模式的方向输出位于 `candidate_NNN/directions/<方向>/`，方向日志为 `candidate_NNN/<方向>.log`。
仍发生异常时保留诊断，不会放宽阈值或将其伪装成抓取通过。

## 7. 汇总、合并与 Isaac Sim 回放

重新汇总已有套件，不重新仿真：

```bash
.venv_l20/bin/python scripts/summarize_l20_isaacsim.py \
  --results data/experiments/l20_extra3_20261104_225733
```

将两个已有实验合并到一个新的目录；重复物体会被拒绝，源文件保留：

```bash
.venv_l20/bin/python scripts/merge_l20_isaacsim_results.py \
  --sources data/experiments/l20_all_acquire_10s_20261104_173323 \
            data/experiments/l20_extra3_20261104_225733 \
  --output "data/experiments/l20_five_objects_merged_$(date +%Y%m%d_%H%M%S)"
```

合并生成 `REPORT.md`、`summary_all.json`、`sources.json`、`index.html` 和回放链接。
合并目录依赖原实验目录，不是可独立搬运的完整副本。

在 Isaac Sim 窗口中回放已完成候选：

```bash
bash scripts/run_l20_isaacsim_replay.sh \
  --results data/experiments/l20_five_objects_acquire_10s_20261105 \
  --object mug --candidate 20 --case gz- --speed 1
```

更换 `--object` 和对应 `--candidate` 可查看其他物体。面板可暂停、切换方向和调整速度。
原生回放使用保存的状态，不重新计算物理；显示 SDF 输入三角网格，并非 SDF 等值面。

### 输出文件

```text
<套件目录>/
  plan.json                 # 生成路径、网格散列、参数和子命令
  status.json               # 套件执行状态
  logs/                     # 生成与验证日志
  <物体>/
    inputs/manifest.json    # 候选、scale、URDF、网格来源、质量和惯量
    inputs/meshes.npz       # 验证用三角网格
    summary.json            # 批量候选结果
    REPORT.md
    candidate_NNN/
      summary.json          # 六方向指标和实际参数
      protocol.json         # 开闭合目标及阶段边界
      trajectory_*.npz      # 实际状态、目标、接触、重力及阶段
      collision_audit.json
      runtime_collision_statistics.json
      runtime_drives.json
  REPORT.md                 # 全部物体统计
  summary_all.json
  index.html
  visualizations/*.html
```

合并目录使用链接，结构与源套件略有不同。完成所有候选表示验证已执行完，不表示抓取全部通过。

## 8. 其他流程与详细文档

当前已新增 **Isaac Sim 残差 PPO**：以某个原始生成位姿为 reference，沿用验证的 SDF 场景、
固定 PD 和物理 mimic，学习16个主动关节的有界、限速残差。训练、续训、六方向零残差/策略配对评估及窗口查看命令见
[残差强化学习说明](docs/ISAACSIM_RESIDUAL_RL.md)。每个参考独立训练，不复用旧 mjlab 检查点。

新实验使用独立的 [残差 PPO v2](docs/ISAACSIM_RESIDUAL_V2.md)：补齐历史/时间观测，严格失败后停止正奖励及学习，
闭合与零重力保持完全跟随原参考，仅重力保持阶段学习残差。v2需新建策略；旧v1入口与检查点保留。
已有v2检查点可使用 `scripts/evaluate_isaacsim_residual_paired.sh` 做每轴±0.5mm的配对位置扰动评估，
两组读取同一份样本并核对显式初态，原训练及检查点散列保持兼容；命令与统计含义见上述v2文档。
下一轮采用独立的 [v3稳定裕量实验](docs/ISAACSIM_RESIDUAL_V3.md)：保留零残差基线，增加原始动作/请求/实际目标残差诊断，
细化低滑移、低转角和小残差奖励；仅提供200批次短训与原扰动表复测流程，v1/v2保持可用。
v3正式复测后策略仅6/30通过、零残差30/30。已通过离线奖励审计定位负存活收益反例，
新增 [v4非负存活质量实验](docs/ISAACSIM_RESIDUAL_V4.md)，保持物理与探索设置不变；其正式复测和新seed结果见持续复盘第14–15节，不能视为全面优于零残差。
失败参考的新阶段见 [先基线复核、再独立训练、同样本评估](docs/ISAACSIM_FAILED_REFERENCE.md)。
积木7已在当前RL环境测得3/6，三个失败均在开启重力后约38–58ms；新入口检查失败阶段与训练准入，避免复用杯子策略或训练不可补救的准备阶段失败。
用户最新指定后续参考为杯子0，操作见 [杯子0残差修复流程](docs/ISAACSIM_MUG0_RRL.md)。原10秒acquire验证4/6，gy+与gz+失败；先在当前RL环境复核，再独立训练和配对评估。杯子28计划已被此选择替代。
杯子20阶段已收尾，版本演进、配对指标及结论边界见 [杯子20 RRL阶段报告](docs/MUG20_RRL_REPORT.md)。
当前优先流程：[独立进程复现、逐指接触诊断与参考细化](docs/ISAACSIM_ISOLATED_REFERENCE_REFINEMENT.md)。已复现杯子0组3的gy+/gz+失败；两个食指微调均无通过数提升，未进入新RRL训练。
杯子0最新实验：[v5预加载残差](docs/ISAACSIM_RESIDUAL_V5.md)已完成训练和两批配对：首批裕量改善，第二批修复2次也破坏2次，完整六方向退化，不作为更优控制器。下一步为预加载幅度消融，见[结果复盘](docs/MUG0_V5_RESULT_ANALYSIS.md)；v4检查点保留。

| 内容 | 入口 |
|---|---|
| Isaac Sim SDF、GUI 与历史协议细节 | [Isaac Sim 文档](docs/ISAACSIM_L20_SDF.md)（含早期 1000 步示例，10 秒使用 2400 步） |
| 五物体 L20：官方 Gym 六方向协议适配与命令 | [Gym 验证文档](docs/ISAACGYM_L20.md)（凸碰撞；mimic 为一致目标，无硬机械耦合） |
| Isaac Sim 原始抓取 reference + 残差 PPO | [Isaac Sim 强化学习](docs/ISAACSIM_RESIDUAL_RL.md)（SDF、物理 mimic、零残差/策略配对评估） |
| MuJoCo / mjlab 细化与残差策略训练 | [mjlab_grasp/README.md](mjlab_grasp/README.md) |
| ShadowHand MuJoCo 与 Gym 协议对照 | [协议对照](assets/shadowhand_official/MUJOCO_GYM_PROTOCOL.md) |
| 早期 MuJoCo 自适应柔顺、ShadowHand / Gym 命令及编译记录 | [历史 README](docs/README_LEGACY_20261105.md) |

查看已有 TensorBoard 曲线，使用项目启动脚本：

```bash
bash scripts/tensorboard.sh
```

浏览器访问 `http://127.0.0.1:6006`。不同引擎的碰撞表示、控制器和验收条件不同，比较结果前需核对协议。

## 引用

本项目基于 [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet)。使用相关代码或数据时请引用原工作：

```bibtex
@article{wang2022dexgraspnet,
  title={DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation},
  author={Wang, Ruicheng and Zhang, Jialiang and Chen, Jiayi and Xu, Yinzhen and Li, Puhao and Liu, Tengyu and Wang, He},
  journal={arXiv preprint arXiv:2210.02697},
  year={2022}
}

@misc{your2026dexgraspnetl20,
  author = {Li Juyang},
  title = {DexGraspNet LinkerHand L20 Extension},
  year = {2026},
  howpublished = {\url{https://github.com/brantleeee/DexGraspNet_LinkerHand_l20}},
  note = {GitHub repository}
}
```

最新生成修正：[v2原始非凸网格查询、五指分组接触及6000步命令](docs/L20_GENERATION_V2.md)。
