# L20：官方 DexGraspNet Gym 协议适配

此入口读取已经验证过的五物体同一批 L20 抓取位姿，直接调用固定版本的官方
`IsaacValidator` 来构造场景和执行测试。原始官方文件不作修改，运行时将其
ShadowHand 关节名表替换为 L20 关节名，加载新准备的 L20 资产。

## 协议与适配边界

保留官方设置：固定手腕、手部关闭重力、物体自由，六个整体旋转构造六个相对重力方向；
世界重力 `(0,-9.8,0)`，dt=1/60 秒、每步两个 substeps，默认 100 步（约 1.67 秒）。
GPU PhysX，位置迭代 8、速度迭代 0，contact offset 10 mm，摩擦系数 3，物体密度 500。
所有转动关节用位置驱动，刚度 1000、阻尼 0，力矩上限来自导入资产，报告保存实际 DOF 属性。
**没有零重力闭合阶段**，从保存的闭合位姿开始测试，不使用当前 Isaac Sim 的 acquire 协议。

输出分两级：

- `simulated_pass`：六方向末帧均有手物接触，且末帧状态数值有效。
- `valid`：上述通过，并且原始 NPY 保存的 `E_pen < 0.001`。

`E_pen` 不会因接触调整而重新计算，也不是仿真最大接触穿透。
不加 5 mm 滑移、15° 转角、10 秒保持或抬升要求。数值异常不计为通过。

必须区分以下 L20 适配与原始官方 ShadowHand 实验：

1. 手与物体均使用 **Gym 凸碰撞**。手部每个导入网格为凸碰撞，物体沿已有网格的连通部件导出，
   交给 Gym 加载；未启用 SDF，不承诺保留凹面细节。
2. Gym 入口没有复用 Isaac Sim 的 PhysX mimic API；导出 URDF 明确移除 mimic 标签，
   使用 **21 个独立位置驱动、目标满足 mimic 关系**。这不是 16 驱动加 5 个硬机械联动，
   实际偏差记录于每方向 `final_max_mimic_error_rad`，仅作诊断，不混入官方通过条件。
3. `official` 接触调整沿用原始的 1 mm 接触阈值、1 mm 法向目标偏移、梯度乘 500，
   但使用 L20 的表面采样，并将从动关节梯度回传到主动关节，再做 URDF 限位和 mimic 投影。
   原始 ShadowHand 调整没有这套 L20 投影，因此应称“官方接触调整的 L20 适配”。
4. `no-force` 完全不做接触调整，直接验证保存的原始位姿。建议两组一起运行，判断调整的影响。

因此这里检验的是“同一批 L20 位姿在官方 Gym 接触筛选协议下的表现”，
不能宣称严格复现官方手型或已经验证 L20 的真实机械联动。若需要被动硬联动一致性，仍需单独实现和核验 Gym 肌腱/约束资产。

## 准备五物体输入

使用 `.venv_l20`，默认读取统一五物体目录。程序核对源 NPY、URDF、网格散列及原始候选下标；
拒绝静默投影旧的原始无效位姿。scale 取保存值，物体缩放一次，手腕位置不另作缩放。
官方调整需要 CUDA / TorchSDF，**此步骤会进行接触梯度计算，但不会运行物理仿真**。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

l20_gym_inputs="data/experiments/l20_gym_inputs_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/prepare_l20_isaacgym.py \
  --results data/experiments/l20_five_objects_acquire_10s_20261105 \
  --objects mug jenga banana bottle camera \
  --variants official no-force --device cuda:0 \
  --output "$l20_gym_inputs"
```

准备文件包含 `manifest.json`、五个 NPZ、手部 URDF / OBJ 和物体 URDF / OBJ。
Manifest 保存输入来源、导出文件散列、联动定义和接触调整量；验证入口会再次检查导出文件散列。
仅测试资产准备而不做 CUDA 接触调整，可使用 `--variants no-force --device cpu`。

## 官方原始 100 步验证

启动脚本自动使用 `.venv_isaacgym`（Python 3.8）和本地 SDK，不使用 Isaac Sim 或 mjlab：

```bash
l20_gym_out="data/experiments/l20_gym_official100_$(date +%Y%m%d_%H%M%S)"

bash scripts/run_l20_isaacgym.sh \
  --inputs "$l20_gym_inputs" \
  --objects mug jenga banana bottle camera \
  --variants official no-force \
  --sim-steps 100 \
  --output "$l20_gym_out"

xdg-open "$l20_gym_out/index.html"
```

五物体 × 16 候选 × 6 方向 × 2 处理组，共 **960 个方向场景**。
只运行 `--variants official` 则为 480 个方向场景。
输出目录必须是新目录；暂未提供断点续跑。各物体/处理组串行，组内候选的六方向并行。
发生程序异常时保留已完成组的报告和 `error.json`，不要使用 Isaac Sim 的续跑命令。

## 原生 GUI 查看一个候选

```bash
bash scripts/run_l20_isaacgym.sh \
  --inputs "$l20_gym_inputs" \
  --objects mug --variants official --indices 20 \
  --sim-steps 100 --gui \
  --output "data/experiments/l20_gym_gui_$(date +%Y%m%d_%H%M%S)"
```

`--indices` 是原始 NPY 下标。窗口默认暂停，空格开始/暂停，1–6 切换方向，
鼠标左键拖动旋转、PageUp/PageDown 缩放、Home 复位；完成后保持窗口，关闭退出。
这是重新执行 GUI 仿真，不是已存轨迹回放。提前关闭不会记为完整验证。
若将 `--sim-steps` 改为 1000，为约 16.67 秒的延长测试，已不属于官方原始时长。

## 结果与已做检查

- `summary.json`：逐物体、处理组、原始候选下标、六方向接触、数值状态、质量、最终 mimic 偏差和实际驱动属性。
- `*_states.npz`：各方向物体初末刚体状态及末帧手部关节状态，不是完整时间轨迹。
- `REPORT.md` / `index.html`：分组统计；HTML 是统计入口，不是三维动画回放。

已完成五物体全部 80 个位姿的 CPU 资产准备与原始数据一致性检查、Gym 环境命令行加载、语法检查。
上述为程序交付时的检查范围；随后用户已运行真实验证，结果见下一节。


## 2026-11-05 实际运行与逐位姿比较

用户完成了五物体、80 个候选、两种处理组的 960 个方向场景，结果位于
`data/experiments/l20_gym_official100_20261105_114721/`。
Gym 原始组最终通过 8/80，调整组通过 9/80；六方向仅接触通过分别为 11/80 和 10/80，
其余差额由源文件 E_pen 筛选排除。

与同批 Isaac Sim 候选的逐位姿、逐方向结果见
[比较页面](../data/experiments/l20_gym_vs_isaacsim_20261105/index.html)。
方向按物体原始坐标统一，已使用全部 960 场景保存的物体初始四元数独立核查方向映射。
`candidates.csv` 包含 80 个候选，`directions.csv` 包含 480 行六方向比较，
同时列出 Gym 原始/调整及 Sim 结果。这里只读取已有输出，没有重新执行仿真。
