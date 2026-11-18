# L20：Isaac Sim SDF 动力学验证

本机使用 `/home/lee/isaac-sim` 的 Isaac Sim 5.1。验证直接使用 PhysX，
不依赖 mjlab，也不通过 Isaac Gym。

**当前默认是 `--protocol acquire`：关闭重力 → 张手后平滑闭合抓取 → 零重力保持 → 开启重力验证。**
此前 32 个候选的统计属于 `direct` 历史协议（直接从闭合位姿开启重力），不能作为新协议的通过率，
批量脚本也禁止跨协议复用结果。

## 先抓取，再开启重力（当前流程）

默认保留保存的手腕位置、手指侧摆与拇指对掌姿态，将屈曲关节朝中性位置张开 60%，
5 个 mimic 关节同步张开。随后在零重力下用五次平滑曲线插值到保存的抓取目标：

1. **2 秒闭合**：物体为自由刚体，重力为 0，手指确实发生闭合动作。
2. **1 秒保持**：仍为零重力，继续保持闭合目标。
3. **开启重力，验证 1000 步**：重力为指定方向的 9.81 m/s²；不重置物体位姿或速度。

开启重力前至少存在一个手–物体接触点才记为已获取接触；最终通过仍需满足重力阶段的稳定性检查。
位移和转角相对开启重力前的实际物体位姿计算，同时单独记录闭合造成的移动；
1 mm 穿透上限检查包括闭合阶段在内的全部采样步。
这不会固定物体、关闭接触或在阶段切换时把物体传送回原点。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

# 实际运行“抓取后开启重力”的六方向验证
bash scripts/run_l20_isaacsim.sh \
  --inputs data/experiments/l20_isaacsim_sdf_all_20261104_1529/mug/inputs \
  --indices 20 --protocol acquire \
  --close-seconds 2 --settle-seconds 1 --open-fraction 0.6 \
  --steps 1000 --sdf-resolution 256 \
  --output "data/experiments/l20_mug_acquire_$(date +%Y%m%d_%H%M%S)"

# 在 Isaac Sim 中回放已完成的三阶段测试，面板会显示当前阶段
bash scripts/run_l20_isaacsim_replay.sh \
  --results data/experiments/l20_isaacsim_acquire_demo_20261104 \
  --object mug --candidate 20 --case gz- --speed 1

# 积木新流程的回放
bash scripts/run_l20_isaacsim_replay.sh \
  --results data/experiments/l20_isaacsim_acquire_demo_20261104 \
  --object jenga --candidate 3 --case gz- --speed 1
```

新协议已实际测试杯子 20、积木 3 各六方向：杯子 **6/6** 个方向通过；积木 **3/6** 个方向通过，
未达到六方向全部通过。这里只测试了两个候选，尚未按新协议重新筛选全部 32 个候选。

- [新协议报告](../data/experiments/l20_isaacsim_acquire_demo_20261104/REPORT.md)
- [新协议离线回放](../data/experiments/l20_isaacsim_acquire_demo_20261104/index.html)

每方向保存 1720 个物理步（闭合 480 + 保持 240 + 重力验证 1000），连同初始帧可回放 1721 帧。
`trajectory_*.npz` 增加 `phase`（0 闭合、1 保持、2 重力验证）、逐步 `gravity`、`joint_targets`、
`contact_count`、初始张手关节角及开启重力前的参考位姿；`protocol.json` 保存阶段边界。
旧直接验证可显式指定 `--protocol direct`。

## 碰撞模型

- 手：`grasp_generation/mjcf/linkerhand_l20_right.urdf` 中的全部 22 个原始 STL，保留三角面，逐 link 使用 SDF。
- 物体：提供的三角网格整体使用 SDF，按保存候选的 scale 缩放一次。
- 不调用 convex hull、convex decomposition 或网格降面。
- 蓝色是手的碰撞网格，橙色是物体的碰撞网格；显示的表面来自 SDF 的输入网格，并非 SDF 等值面的重建。
- `collision_audit.json` 记录 SDF 参数、网格散列和网格间距；`runtime_collision_statistics.json` 检查 PhysX 中有 23 个 triangle-mesh shape、0 个 convex shape。不满足即停止。

**现有物体资源的限制：**项目本地杯子和积木只有恢复的 `coacd/decomposed.obj`。
将它们改成 SDF 不会恢复之前分解丢失的细节。手部使用完整 STL；物体仍沿用现有几何来源。
若有原始物体网格，在准备命令加入 `--object-mesh /路径/original.obj`；必须保持抓取生成时的
归一化坐标、原点和方向，不能直接用另一个杯子模型替换。准备阶段记录 watertight 状态；
非封闭输入的 SDF 内外判断仍需结合接触和可视化检查。

## 单个候选

在仓库根目录执行。`--indices` 是 NPY 中的原始数组下标；杯子 scale 0.06 的组内候选 7
对应原始下标 18。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

.venv_l20/bin/python scripts/prepare_l20_isaacsim.py \
  --object mug --scale 0.06 --indices 18 \
  --output "data/isaacsim_l20/mug_$(date +%Y%m%d_%H%M%S)"
```

本次已准备的输入可以直接复用：

```bash
# 积木；鼠标可调整视角，结束后保持窗口，关闭窗口退出
bash scripts/run_l20_isaacsim.sh \
  --inputs data/isaacsim_l20/jenga_006_7 \
  --cases gz- --steps 1000 --sdf-resolution 256 \
  --gui --snapshots --viewer-seconds -1 \
  --output "data/experiments/l20_jenga_sdf_gui_$(date +%Y%m%d_%H%M%S)"

# 杯子
bash scripts/run_l20_isaacsim.sh \
  --inputs data/isaacsim_l20/mug_006_18 \
  --cases gz- --steps 1000 --sdf-resolution 256 \
  --gui --snapshots --viewer-seconds -1 \
  --output "data/experiments/l20_mug_sdf_gui_$(date +%Y%m%d_%H%M%S)"
```

去掉 `--cases gz-` 即测试六个重力方向。去掉 `--gui --viewer-seconds -1` 即无窗口运行。
`--snapshots` 可在无窗口时保存初始及各方向结束的 PNG。首次烘焙 SDF 较慢，
`status.json` 显示当前进度；后续同一网格通常可以复用 PhysX 缓存。
启动脚本通过进程锁串行运行本项目验证，避免多个 RTX 管线同时编译导致桌面无响应。
图形界面每个循环只推进一个物理步，再单独刷新画面；渲染帧数不代替物理步数。

默认物理步长为 **1/240 秒**，1000 步为每方向 **4.167 秒**；
这里的 1000 步指重力验证阶段；默认 acquire 另含 3 秒获取阶段，总时长 **7.167 秒**。
要测试与 Gym 1000 × 1/60 秒相同的物理时长，指定 `--steps 4000`。
这不表示两个引擎的求解器和接触算法相同。

## 批量验证某个比例下的全部候选

```bash
.venv_l20/bin/python scripts/batch_l20_isaacsim.py \
  --object jenga --scale 0.06 --steps 1000 --sdf-resolution 256 \
  --output "data/experiments/l20_jenga_sdf_all_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/batch_l20_isaacsim.py \
  --object mug --scale 0.06 --steps 1000 --sdf-resolution 256 \
  --output "data/experiments/l20_mug_sdf_all_$(date +%Y%m%d_%H%M%S)"
```

可加 `--indices 0 7 12` 选择原始数组下标，或 `--result-file /路径/新结果.npy`
使用新的生成结果。不会对不同 scale 的手腕位置作猜测性缩放：选择的候选比例不匹配时直接报错。
各候选独立启动进程，避免 SDF 碰撞和 articulation 缓存跨候选污染。

## 控制和判定

手腕固定在保存的抓取位姿，物体是自由刚体，无地面、无固定约束和外部支撑。
21 个关节包含 16 个独立位置驱动及 5 个 PhysX mimic 关节。使用 URDF 的质量、惯量、
关节角度范围及驱动力矩上限。固定 PD 默认 `kp=5`、`kd=0.05`，摩擦系数 3；
手和物体均受重力，手部自碰撞关闭，与此前默认 L20 验证设置一致。
此处没有接入训练策略或自适应柔顺控制。

原生 PhysX `maxJointVelocity` 在本机此 articulation + mimic 配置中实测会造成 NaN，
目前未启用关节速度上限；`runtime_drives.json` 明确记录这一点。
URDF 中的角度限位和力矩上限仍启用。

每方向从相同抓取状态重置。通过要求采样物理步的最大位移不超过 5 mm、最大转角不超过
15°、接触报告最大穿透不超过 1 mm，末尾连续 1 秒接触力超过 0.001 N，且数值有限。
穿透来自 PhysX 报告的接触点，不是对全部三角网格做精确最大穿透计算，也不包含步进前的静态穿透审计。
六方向必须全部通过才将候选标记为通过；这是本项目的严格标准，不等同于官方 Gym 的末帧有接触标准。

保存 `scene.usda`、`trajectory_*.npz`、`results.json`、`summary.json`、`REPORT.md`，
以及碰撞审计和运行时驱动参数。数值失败保存在 `error.json`，启动脚本返回非零退出码。

## 2026-11-04 实测与 GUI 状态

最终使用串行、无窗口、无截图运行；每方向 1000 步，SDF resolution 256。
两组共 12 个试验执行完成，实际 PhysX 驱动读回 kp=5、kd=0.05，
每组统计均为 23 个 triangle-mesh shape、0 个 convex shape。

|物体|原始候选下标|通过方向|六方向全部通过|
|---|---:|---:|---|
|杯子 scale 0.06|18（组内 7）|5/6|否；gz- 位移 6.15 mm|
|积木 scale 0.06|7|3/6|否|

最终报告：

- [杯子](../data/experiments/l20_isaacsim_sdf_mug_serial_20261104/REPORT.md)
- [积木](../data/experiments/l20_isaacsim_sdf_jenga_serial_20261104/REPORT.md)

**上述 serial 目录是初期单候选的完成结果；全部候选结果以下一节批次为准。**之前文件名含 final、v2 的目录仍属于调试过程：
部分运行的 USD 角度增益未换算，实际 kp 约 286；pd5 与 GUI 测试则在此前会话中断。
这些目录不应与最终结果混用。

GUI 曾在 RTX pipeline 编译阶段持续等待超过 190 秒，日志为
`Waiting for RtPso async group async compilation`，并发进程加重了资源争用。
已加入串行锁、限制 CPU 线程，并将 GUI 的物理步进和渲染刷新分开。
无窗口模式已复测完成；GUI 卡顿修复尚未复测通过，打开 GUI 仍可能等待首次 RTX 编译。

## 全候选验证与离线回放

2026-11-04 的全候选批次位于
`data/experiments/l20_isaacsim_sdf_all_20261104_1529`，已完成杯子、积木各 16 个 scale 0.06 候选，共 192 个方向试验。
原杯子 18、积木 7 的已完成结果经过输入散列、SDF 审计、实际驱动增益、步长及判定阈值核对后复用，
其余 30 个候选依次运行。批次状态见 `status.json`，完整统计见 `REPORT.md` 和 `summary_all.json`。

结果：杯子 **9/16** 个候选通过六方向，原始数组下标为 **0、4、5、6、20、24、27、28、30**；
积木 **0/16** 个候选通过六方向。全部试验数值有效，运行时均为 23 个 triangle-mesh shape、0 个 convex shape。
每方向物理时长为 4.167 秒；通过结果仅对应本批次记录的几何、摩擦和控制参数。

- [全候选报告](../data/experiments/l20_isaacsim_sdf_all_20261104_1529/REPORT.md)
- [全部离线回放](../data/experiments/l20_isaacsim_sdf_all_20261104_1529/index.html)

离线回放使用保存的 Isaac Sim 关节状态和物体位姿，支持鼠标旋转/缩放、切换六方向、播放/暂停、
拖动时间轴、分别显示手与物体、查看网格线；不启动 Isaac Sim 或 RTX，也不重新进行物理计算。
页面内包含全部模型和固定版本 Three.js，无需联网。显示的仍是 SDF 输入三角面，不是 SDF 等值面。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

# 打开已导出的全部候选可视化目录
xdg-open data/experiments/l20_isaacsim_sdf_all_20261104_1529/index.html

# 单独导出某个已完成候选的离线回放
.venv_l20/bin/python scripts/replay_l20_isaacsim_html.py \
  --results data/experiments/l20_isaacsim_sdf_all_20261104_1529/mug/candidate_000 \
  --inputs data/experiments/l20_isaacsim_sdf_all_20261104_1529/mug/inputs \
  --output data/experiments/l20_isaacsim_sdf_all_20261104_1529/mug/candidate_000/replay.html

# 汇总完成结果，并导出每个候选的离线回放
.venv_l20/bin/python scripts/summarize_l20_isaacsim.py \
  --results data/experiments/l20_isaacsim_sdf_all_20261104_1529
```

若需要在新批次中复用相同参数的单候选完成结果，批量脚本支持
`--reuse-results /路径/单候选结果目录`。参数、源数据或模型不一致时直接拒绝复用。

## Isaac Sim 原生回放

原生 GUI 回放已实际测试通过。它将保存的关节与物体状态写成 USD 动画（旧 direct 为 1001 帧，新 acquire 为 1721 帧），
使用与验证相同的网格，停用回放副本的刚体、碰撞计算、关节约束和物理场景，
不重新进行动力学验证或 SDF 烘焙，也不修改原始验证场景。
显示的是 SDF 输入网格，不是 SDF 等值面。图形播放依然需要 Isaac Sim 的 RTX 渲染。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

# 杯子，原始数组下标 20，正常速度循环回放
bash scripts/run_l20_isaacsim_replay.sh \
  --object mug --candidate 20 --case gz- --speed 1

# 积木；关闭前一个窗口后再运行
bash scripts/run_l20_isaacsim_replay.sh \
  --object jenga --candidate 3 --case gz- --speed 1
```

窗口中的 **L20 Recorded Grasp Replay** 面板提供 Play/Pause、Restart、Reset camera、
速度滑条、时间滑条、六方向下拉框，以及分别显示 Hand/Object 的复选框。
视口使用 Isaac Sim 自带相机导航；关闭窗口退出。默认循环播放，
加 `--paused` 可以从初始帧暂停开始，加 `--no-loop` 可以在播完后停住。
`--speed 1` 对应记录的实际时长（旧 direct 为 4.167 秒，新 acquire 为 7.167 秒）；加载与着色器编译时间不计入回放时间。
默认数据目录现为新三阶段示例 `data/experiments/l20_isaacsim_acquire_demo_20261104`。
查看历史 32 个 direct 候选时，显式加
`--results data/experiments/l20_isaacsim_sdf_all_20261104_1529`。

回放入口与动力学验证入口共用进程锁，保持单个 Isaac Sim 实例运行。
`data/isaacsim_l20/replay_*.json` 记录启动、动画构建、当前方向、播放进度与退出状态。

可选导出独立 USD 动画（只包含命令指定方向）：

```bash
bash scripts/run_l20_isaacsim_replay.sh \
  --object mug --candidate 20 --case gz- \
  --export-usd "data/experiments/mug20_animation_$(date +%Y%m%d_%H%M%S).usda"
```

本机原生 GUI 测试记录：
`data/experiments/l20_isaacsim_native_replay_check_20261104/status_gui.json`。
该测试正常关闭，读取 1001 帧、原轨迹时长 4.167 秒、保留 23 个 SDF 输入网格，物理计算关闭。

## 新增三个物体：香蕉、瓶子、相机

物体均已在本机 `data/meshdata` 中找到，不需要下载。统一 scale 0.06 时：

|名称 / 命令参数|物体编号|尺寸约 mm|密度 500 kg/m³ 下质量约 g|
|---|---|---|---:|
|香蕉 `banana`|`ddg-gd_banana_poisson_002`|90.85 × 78.86 × 19.96|15.17|
|瓶子 `bottle`|`sem-Bottle-437678d4bc6be981c8724d5673a063a6`|32.21 × 32.21 × 113.97|30.57|
|相机 `camera`|`sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5`|98.15 × 44.81 × 59.74|79.10|

这些物体的源几何仍是从项目可视化恢复的旧 CoACD 网格；Isaac Sim 对其整体建立 SDF，
没有新增凸包或凸分解碰撞体，但不能恢复原始模型细节。共享配置位于
`scripts/l20_object_catalog.py`，准备、批量验证、原生回放和统计均支持这三个名称。

随项目附带的旧位姿各有 128 个，其中 scale 0.06 各 28 个，**这 28 个全部不满足当前 L20 URDF 约束**。
因此新模型必须提供新生成的 `--result-file`；准备脚本拒绝将这些旧位姿强行投影后当成新位姿测试。
下面的入口先重新生成与 scale 对应的位姿，再进行验证。

### 一条命令生成并验证三个模型

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

l20_out="data/experiments/l20_extra3_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/run_l20_object_suite.py \
  --objects banana bottle camera \
  --scale 0.06 --batch-size-per-scale 16 --n-iter 6000 \
  --close-seconds 2 --settle-seconds 1 --open-fraction 0.6 \
  --gravity-seconds 10 --sdf-resolution 256 \
  --output "$l20_out"

# 全部执行完成后，在同一个终端打开可视化目录
xdg-open "$l20_out/index.html"
```

每个模型新生成 16 个 L20 位姿，共 48 个候选、288 个方向试验。
生成采用现有 L20 优化器、6000 次迭代和 scale-aware 初始化；每个方向独立执行
零重力闭合 2 秒、零重力保持 1 秒、开启重力保持 10 秒。Isaac Sim 碰撞使用 SDF 256，
保持原来的固定 PD 5 / 0.05、摩擦系数 3 和严格判定阈值。生成与验证均串行运行。

`plan.json` 保存网格散列、比例、尺寸、生成结果路径及实际子命令，
`logs/<物体>_generate.log` 和 `logs/<物体>_validate.log` 保存运行输出；
每个物体目录有逐候选报告和完整轨迹。全部结束后自动导出根目录的
`REPORT.md`、`summary_all.json`、`index.html` 及 `visualizations/*.html`。
运行脚本不自动重复失败任务，也不覆盖已有实验目录。

只检查计划、不启动任何 GPU 任务：

```bash
.venv_l20/bin/python scripts/run_l20_object_suite.py \
  --objects banana bottle camera --scale 0.06 \
  --output "$l20_out" --plan-only
```

### 单独验证已有的新生成位姿

如需分开执行，从 `plan.json` 取出该物体的 `result_file`，运行：

```bash
.venv_l20/bin/python scripts/batch_l20_isaacsim.py \
  --object banana --scale 0.06 \
  --result-file /绝对路径/新生成的/ddg-gd_banana_poisson_002.npy \
  --protocol acquire --close-seconds 2 --settle-seconds 1 --open-fraction 0.6 \
  --steps 2400 --sdf-resolution 256 \
  --output "data/experiments/banana_validate_$(date +%Y%m%d_%H%M%S)"
```

全部结束后，也可手动导出三物体统计与离线回放：

```bash
.venv_l20/bin/python scripts/summarize_l20_isaacsim.py --results "$l20_out"
```

原生 Isaac Sim 回放示例（全部流程执行结束后）：

```bash
bash scripts/run_l20_isaacsim_replay.sh \
  --results "$l20_out" --object banana --candidate 0 --case gz- --speed 1
```

本次新增支持已检查模型读取、尺寸/质量、三物体汇总发现、参数计划和旧位姿拒绝逻辑；
没有启动这三个物体的长时间位姿生成或动力学测试，故目前没有它们的新协议成功率。

### 已有实验失败后的续跑

若某个候选出现 `Nonfinite physics state`，它是仿真数值异常，不能当作普通抓取失败或通过。
其余完成候选仍可独立汇总，命令见上文 `summarize_l20_isaacsim.py`，报告会标明未完成项。

```bash
.venv_l20/bin/python scripts/run_l20_object_suite.py \
  --output data/experiments/l20_extra3_20261104_225733 --resume
```

`--resume` 使用原 `plan.json` 中的参数和生成位姿，不重新生成。
重新检查当前源位姿、URDF、网格及已完成候选的协议、碰撞和驱动审计，
复用完成候选；仅重新执行未完成候选。旧失败候选目录和日志移入对应物体目录的
`failed_attempts/<时间戳>/`。续跑结束或再次失败时均尝试导出已有结果和 HTML。
批处理也可单独使用 `--resume`，其余参数必须与原运行一致。

数值异常的物理根因尚不能仅凭日志确定；续跑保留相同物理参数和验收标准，
不保证重试会消除异常。新增 `numerical_failure.json` 记录异常方向、阶段、精确步数及非有限状态字段，供再次失败时定位。

### 重复数值发散：隔离六个方向的仿真进程

相机候选 3 的实际重试在 `gy+`（第四个方向）、零重力闭合第 60 步出现
非有限关节角；此时还没有开启重力。原流程在同一 PhysX 场景中切换方向，
只重设关节与物体状态，并不能证明所有求解器内部状态都已清除。
这提供了检查跨方向状态影响的线索，尚不能证明它是数值异常的根因。

使用独立进程排除此因素：

```bash
.venv_l20/bin/python scripts/run_l20_object_suite.py \
  --output data/experiments/l20_extra3_20261104_225733 \
  --resume --isolate-cases
```

仅对未完成候选执行六次独立启动，每次启动只验证一个方向；已完成候选继续审计复用。
零重力闭合/保持、重力时长、SDF、驱动参数和通过阈值不变。
单方向输出和日志保存在 `candidate_003/directions/<方向>/` 与
`candidate_003/<方向>.log`。六方向均执行完并通过数值、碰撞和驱动审计后，
才合并为标准候选输出；普通抓取未通过仍会正常记录，不会被伪装成通过。
任何方向再次数值异常都会保留为未完成，不计入有效结果。
根目录汇总会标明哪些候选改用了独立进程，原报告与回放路径保持兼容。

已通过无仿真的合并/失败分支检查。独立进程增加 Isaac Sim 启动开销，
是否消除本候选的异常仍需实际运行确认。
