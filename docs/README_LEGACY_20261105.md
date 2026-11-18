# DexGraspNet-Main shadowhand to linkerhand l20

## 本机快速运行（RTX 5070 系列）

在项目根目录使用项目环境 `.venv_l20`。该环境复用本机
`/home/lee/miniconda3/envs/UniVTAC` 的 Python 3.11 和 PyTorch 2.7.0+cu128，
并单独安装本项目依赖。原 `environment.yml` 的 Python 3.7 / CUDA 11.3
环境不适用于本机的 Blackwell GPU。

### 小尺寸物体：按比例重新生成并分别保存

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv_l20/bin/python scripts/generate_small_grasps.py \
  --scales 0.04 0.06 --batch-size-per-scale 16 --n-iter 6000 \
  --name "l20_small_$(date +%Y%m%d_%H%M%S)"
```

`scale` 是**原始归一化 mesh 顶点的线性乘数**，不是将已经缩放的物体再次缩放。
此前展示的杯子为 scale=0.1；0.04 和 0.06 分别是该杯子线性尺寸的 40%、60%。
手部 mesh 不缩放，每个候选在其对应尺寸的物体周围重新初始化、重新优化。

**初始化方法区分：** [DexGraspNet 原论文第三节](https://tengyu.ai/assets/pdf/ICRA23_DexGraspNet.pdf)
有意将物体凸包向外扩张 0.2 m，再采样并随机偏移手部初始位姿。
直接运行 `grasp_generation/main.py` 的 `--hull-inflation paper`（默认）保留这一设置。
本节 `generate_small_grasps.py` 默认使用 `--hull-inflation scale-aware`，将扩张距离设为
`clip(0.3 × 当前缩放物体最短外包边长, 0.005, 0.02)` m；这是针对小物体的 L20 实验改动，
不是原方法的错误修复。可传 `--hull-inflation paper` 做同参数对照。
该选择只改变初始化，不能单独证明最终抓取稳定。

2026-10-30 积木对照：`2_of_Jenga_Classic_Game`、scale 0.06、16 个候选、
1000 次优化，物体外包尺寸约 31.2 × 33.6 × 110.4 mm。
原论文 0.2 m 初始化的结果在 `data/experiments/l20_jenga_006_screen_20261030/`；
缩放相关初始化的结果在 `data/experiments/l20_jenga_006_near_surface_20261030/`。
两组都只有 0/16 个候选同时满足 `E_fc < 0.3`、`E_dis < 0.005 m`、`E_pen < 0.001`。
使用相同的 MuJoCo 自适应控制、摩擦系数 3、`gz-` 与抬升测试，
两组各 32 个场景均为 0 通过。旧组同控制器结果保存在
`data/experiments/l20_jenga_006_screen_20261030/dynamics_mu3_same_controller/`。
新组中 `gz-` 最小相对滑移为 5.52 mm
（候选 14），抬升最小滑移为 40.59 mm；因此这次初始化改动没有产生稳定抓取。
这些数据只覆盖 1000 次优化，不能推断 6000 次优化的结果。

对新组候选 14 另测六个轴向重力方向，六项均未达到当前验证器的
滑移 < 5 mm、旋转 < 15° 且有接触的通过条件；接触最大穿透为
1.22–1.81 mm，超过原论文的 1 mm 筛选阈值，抬升滑移为 40.59 mm。
[原论文第三节](https://tengyu.ai/assets/pdf/ICRA23_DexGraspNet.pdf)
是在 Isaac Gym 中用位置控制目标、六方向重力和穿透阈值做物理筛选。
这里的 MuJoCo 测试包含六方向与穿透记录，但自适应控制器、接触链节目标设定、
测试时长和仿真引擎均与原论文不同，因此是本项目的物理检验，不是原筛选协议的完整复现。
数值见 `data/experiments/l20_jenga_006_near_surface_20261030/dynamics_six_axis_mu3/results.json`。

| 杯子 scale | X 外包尺寸 | Y 外包尺寸 | Z 外包尺寸 |
|---|---:|---:|---:|
| 0.04 | 44.4 mm | 49.8 mm | 61.9 mm |
| 0.06 | 66.6 mm | 74.7 mm | 92.8 mm |

输出包含完整候选 `.npy`，以及按比例分组的 `results/scales/scale_0p04/`、
`results/scales/scale_0p06/`，各目录有独立的对象 `.npy` 和 `_vis_best.html`。
每个候选的 `scale` 与 `qpos` 成对保存，`source_index` 保留其原始索引。
`qpos` 平移单位为米，已经是对应比例下优化出的世界位姿，**不能再次乘物体比例**；
Euler 角和关节角单位为 rad，也不能缩放。
`*_scales.json` 保存原 mesh 指纹、每个比例的尺寸、候选数量及分组映射；
`scale_validation.json` 保存逐比例的约束与几何阈值检查。
可视化和动力学入口会核对原 mesh 指纹，防止修改 OBJ 后又叠加 `scale`。

直接使用原入口也可以显式指定比例（混合批次中均衡分配）：

```bash
.venv_l20/bin/python grasp_generation/main.py --name small_custom \
  --object-scales 0.04 0.06 --batch_size 32 --n_iter 6000
```

MuJoCo 可直接读取按比例保存的结果，自动选择该文件的最佳候选。
也可读取完整候选文件并添加 `--scale 0.04`，只在保存比例为 0.04 的候选中选择，
不会把旧位姿重新缩放。`--indices` 是当前输入文件内的局部索引，
省略时自动选择，可避免将旧实验的第 16 个索引用于只有 16 个条目的分组文件。

2026-10-30 已完成 `data/experiments/l20_small_20261030/`：两种比例各 16 个候选，
均优化 6000 步；0.04 有 1/16、0.06 有 4/16 通过几何阈值，选中组内索引分别为 14、7。
32 个初始和最终位姿均满足限位及 DIP 联动，分组导出没有改变世界位姿。
每组所选候选的动力学验证仍为 0/14 通过，不能将几何通过视为稳定抓取。
比例、尺寸、质量和位姿的详细核验见该实验的 `REPORT.md`、`VALIDATION.json`。

```bash
# 在项目根目录执行，实验名可省略（默认使用当前时间，已有实验不会覆盖）
bash scripts/run_local_example.sh
# 可显式指定实验名、迭代次数、候选数（较小批量可减少计算时间）
bash scripts/run_local_example.sh l20_grasp_example 6000 32
# 仅检查程序是否能执行时，才使用 10 次迭代
bash scripts/run_local_example.sh l20_smoke_example 10 128
```

该脚本默认运行杯子示例的 128 个抓取候选、6000 次优化迭代，再导出
`data/experiments/<实验名>/results/` 下的 `.npy` 和 `_vis_best.html`。
可视化优先选择同时满足 `E_fc < 0.3`、`E_dis < 0.005`、`E_pen < 0.001`
的候选，再按总能量排序。没有满足阈值的候选时会明确输出警告，
不能将 HTML 成功导出视为已得到有效抓取。
默认只显示最终手姿；给 `tests/visualize_result.py` 添加 `--show_initial` 可叠加粉色初始手姿，
添加 `--num 0` 可明确查看第 0 个候选。10 次迭代仅用于程序检查，通常仍远离杯面。

2026-10-29 本机验证：`l20_local_20261029` 已生成 128 个有限数值的抓取候选，
TensorBoard 记录了第 0–10 步，平均能量从 272.83 降至 173.04。
`l20_batch_local_20261029` 已通过 6 个物体的 `--all` 小批量生成及 HTML 可视化
（每物体 2 个候选、10 次迭代）。结果保存在对应的 `data/experiments/` 子目录，
数值检查和环境版本分别记录在 `validation.json` 和单物体实验的 `environment.json` 中。

完整抓取验证保存在 `data/experiments/l20_grasp_20261029/`：32 个候选运行 6000 步，
13 个通过上述阈值，默认选中第 16 个。其 `E_fc=0.0281`、`E_dis=0.000997 m`
（4 个优化接触点平均约 0.25 mm）、`E_pen=0.000307`。
实际网格距离检查确认五根手指各有距杯面不足 1 mm 的接触候选点；
`results/grasp_preview.png` 为该最终姿态的浏览器截图。
上述仅是几何贴合结果。后续动力学验证**未通过**：第 16 个候选违反关节限位和
5 个 DIP 联动约束；按 URDF 修正后，六方向保持及抬升测试均失败。
完整记录见 `data/experiments/l20_grasp_20261029/dynamics_hand_object/REPORT.md`。

### 动力学验证（MuJoCo）

最初本机未安装旧版 Isaac Gym，因此新增独立 MuJoCo 验证入口；现在也已安装并实测
Isaac Gym，使用下文独立入口。两种引擎的结果分别记录。
从 URDF 导入惯性、限位、力矩上限及 5 个 mimic 联动，使用 27 个杯子凸分解部件。
杯子为自由刚体，无焊接、桌面支撑或附着力。默认测试摩擦系数 1 和 3，
每组包括六方向重力保持及 10 cm 手腕抬升，记录相对滑移、转角、接触力和驱动力矩。

```bash
.venv_l20/bin/python -m pip install mujoco==3.17.0
# 默认验证候选 16；已有输出目录不会覆盖，请使用新的目录名
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/validate_grasp_dynamics.py \
  --controller fixed \
  --output data/experiments/l20_grasp_20261029/dynamics_repeat
# 渲染上述已记录轨迹（需要可用的 EGL 环境），视频为 2 倍慢放
MUJOCO_GL=egl .venv_l20/bin/python scripts/render_grasp_dynamics.py \
  data/experiments/l20_grasp_20261029/dynamics_repeat/candidate_016_mu3_lift_fixed.xml
```

`results.json` 中分别记录原姿态是否满足 URDF 约束、修正姿态的动力学结果，
不会把修正后姿态的测试结果当作原抓取通过。标准手-杯测试关闭手部自碰撞，
可添加 `--self-collision` 做敏感性检查；摩擦、密度及控制器增益尚未经过实机标定。
旧版生成器将 5 个联动 DIP 当成独立变量，并只以软能量惩罚关节越界，
因此 2026-10-29 的 32 个候选均不满足完整 URDF 约束。
新版生成时已硬约束关节限位，将 DIP 梯度合并至独立关节，并按联动公式更新；
保存仍使用兼容 URDF 的 21 个关节角，其中实际仅有 16 个独立优化自由度。
旧结果文件不作静默修改，需要重新生成。

### 关节自适应柔顺控制

验证入口默认使用 `adaptive`，也可通过 `--controller fixed adaptive` 对比两种控制。
核心控制器为 `grasp_generation/utils/adaptive_compliance.py`，MuJoCo 接口为
`grasp_generation/utils/mujoco_compliance.py`，仅驱动 16 个独立关节，5 个 DIP 保持被动联动。
驱动力矩采用带力矩上限的关节阻抗：

```text
tau = clip(K(t) * (q_reference(t) - q) - D(t) * q_velocity, -effort, +effort)
```

根据每指法向接触力、跟踪误差和速度调整刚度；过载时降低刚度并有限退让，
接触不足时有限速地增加 MCP pitch/PIP 的闭合量。阻尼根据
`diag(T.T @ M(q) @ T)` 的等效惯量、刚度及负载调节，包含 DIP 联动反射惯量。
有接触力低通滤波、增益变化率限制、目标偏移限幅、饱和/跟踪误差防积分累积。
目标角范围和目标角速度同时考虑独立关节及其联动 DIP 的 URDF 约束。
实际关节受冲击时的速度不作硬截断。掌根接触只计入总接触力，不混入每指反馈。
目标接触力是阻抗调节参考，不代表已实现精确力跟踪或已证明变阻抗系统的被动性。

默认参数：每指目标力 2 N，其余手指刚度 0.25–5 Nm/rad，
中指三个主动关节的刚度乘以 1.2（0.3–6 Nm/rad）；
中指 MCP pitch/PIP 闭合参考速度由 0.25 提至 0.375 rad/s，
中指 MCP roll 不执行闭合偏移。阻尼 0.015–0.2 Nm·s/rad，
闭合目标偏移最多 +0.25 rad、退让最多 0.15 rad，驱动力矩按 URDF 限制（当前为 1 Nm）。
参数可编辑 `grasp_generation/configs/adaptive_compliance.json`，或用 `--target-force` 指定目标力。
这些参数作用于原生 MuJoCo 自适应控制测试，不会改变已经训练的 mjlab 策略或其 XML 固定增益。

```bash
# 同一抓取、初始状态、质量、摩擦和加载过程的固定/自适应对比
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/validate_grasp_dynamics.py \
  --controller fixed adaptive \
  --compliance-config grasp_generation/configs/adaptive_compliance.json \
  --output data/experiments/l20_grasp_20261029/compliance_repeat
# 只运行自适应抬升测试
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/validate_grasp_dynamics.py \
  --controller adaptive --cases lift --friction 3 --target-force 2 \
  --output data/experiments/l20_grasp_20261029/adaptive_lift_repeat
```

每个场景额外生成 `_joints.csv`，记录逐关节角度、速度、目标、刚度、阻尼、
滤波后的每指接触力和实际电机力矩；`.npz` 同时保存增益及目标轨迹。
本机对比见 `data/experiments/l20_grasp_20261029/compliance_comparison/REPORT.md`：
4 项控制器测试和 28 个仿真场景的边界/力矩方程核验通过；
摩擦系数 3 的抬升中，峰值电机力矩从 0.252 降至 0.146 Nm（约 42%），
但固定、自适应两组均为 0/14 项抓取通过，接触峰值和滑移没有改善。
自适应柔顺不能替代原抓取的机械约束修正；本实现接入的是本地仿真，未接入实机驱动。

### 带窗口测试稳定性，并显示双方碰撞 mesh

在项目根目录执行以下命令。窗口默认暂停，先检查初始碰撞形状，再按 **F8** 开始：

```bash
MUJOCO_GL=glfw OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/validate_grasp_dynamics.py \
  --viewer --indices 16 --controller adaptive --cases lift --friction 3 \
  --hand-mesh collision --object-mesh collision --speed 0.25 \
  --output "data/experiments/mujoco_view_$(date +%Y%m%d_%H%M%S)"
```

默认同时显示 **22 个蓝色手部碰撞凸包**、**27 个分色杯子碰撞部件**，
关闭详细外观 mesh，启用 MuJoCo 的凸包显示标志，避免把详细 STL 表面误当作碰撞形状。
手部外观、杯子外观、手部碰撞、杯子碰撞分别位于 geom group 1、2、3、4。

- **F5**：手部 `collision → visual → both → none` 循环。
- **F6**：杯子按相同顺序独立切换。
- **F7**：切换接触点、接触力显示。
- **F8**：暂停/继续。鼠标可旋转、平移、缩放观察视角。
- `--hand-mesh none`：仅显示杯子；`--object-mesh none`：仅显示灵巧手。
- `--no-start-paused` 自动开始；`--no-keep-open` 完成后自动关闭。

隐藏 mesh 只影响画面，不关闭物理碰撞。杯子没有桌面支撑或固定约束，
测试在 0.3 秒无重力接触调整后施加重力，并在 1 秒内将手腕抬升 10 cm。
`results.json`、CSV、NPZ 保存到 `--output` 指定目录。提前关闭窗口会取消测试，
写入 `cancelled.json`，不会误记为通过；GUI 模式一次只运行一个候选/摩擦/控制器/场景。

不打开窗口时可运行完整六方向及抬升对比：

```bash
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/validate_grasp_dynamics.py \
  --indices 16 --controller fixed adaptive \
  --output "data/experiments/mujoco_batch_$(date +%Y%m%d_%H%M%S)"
```

录制/回放已保存的碰撞 mesh 视频（`<测试目录>` 替换为实际路径）：

```bash
MUJOCO_GL=egl .venv_l20/bin/python scripts/render_grasp_dynamics.py \
  <测试目录>/candidate_016_mu3_lift_adaptive.xml \
  --hand-mesh collision --object-mesh collision
```

已验证窗口可正常打开、运行和退出，碰撞显示切换不改变质量/碰撞掩码，
且确实选择凸包渲染数据。现有候选 16 的动力学结果仍为 FAIL，不能把窗口显示成功视为抓取成功。

仓库未附带完整 `data/meshdata`。脚本会调用
`scripts/restore_example_mesh.py`，从仓库自带的杯子 HTML 中提取物体网格，
按对应抓取的 scale 还原到归一化坐标，并保存来源信息；已有网格不会覆盖。
这仅恢复示例使用的网格，没有重新进行 CoACD 分解。本机已恢复仓库自带的 6 个物体；
使用其他物体前仍需准备对应模型。

本机环境的安装过程（需要 CUDA Toolkit 12.8 和 C++ 编译器）：

```bash
/home/lee/miniconda3/envs/UniVTAC/bin/python -m venv --system-site-packages .venv_l20
.venv_l20/bin/python -m pip install plotly urdf-parser-py transformations fvcore iopath ninja
.venv_l20/bin/python -m pip install --no-build-isolation -e thirdparty/pytorch_kinematics
# Ninja 必须在 PATH 中，否则会退回速度较慢的串行编译
export PATH="$PWD/.venv_l20/bin:$PATH"
git clone --depth 1 --filter=blob:none --no-checkout --branch v0.7.9 \
  https://github.com/facebookresearch/pytorch3d.git /tmp/l20-pytorch3d-source
git -C /tmp/l20-pytorch3d-source sparse-checkout init --cone
git -C /tmp/l20-pytorch3d-source sparse-checkout set pytorch3d projects/implicitron_trainer
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/usr/local/cuda \
  .venv_l20/bin/python -m pip install --no-build-isolation \
  /tmp/l20-pytorch3d-source
IGNORE_TORCH_VER=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/usr/local/cuda \
  .venv_l20/bin/python -m pip install --no-build-isolation thirdparty/TorchSDF
```

<figure>
<img src="README-media/4aa94486bc7ba4d85601eca66b954e56ad2781c4.png"
class="wikilink" alt="Pastedimage20260510122737.png" />
<figcaption
aria-hidden="true">Pastedimage20260510122737.png</figcaption>
</figure>

    python main.py --name l20_05101215 --object_code_list core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03 --n_contact 4 --n_iter 6000

    python tests/visualize_result.py   --object_code core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03   --result_path ../data/experiments/l20_05101215/results

<img src="README-media/d84ba51206fb7fec5f4fbd8d8466d14f818b7628.png"
class="wikilink" alt="Pastedimage20260510123411.png" />
\## 运行检查shdowhand对比


    cd ~/gpufree-data/DexGraspNet-main/grasp_generation

    python main.py --name main05021t --object_code_list core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03 --gpu "0"

    python tests/visualize_result.py \
      --object_code core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03 \
      --result_path ../data/experiments/main05021t/results

<figure>
<img src="README-media/561393187a93ba964a950a00bbac46a9c17c1d5e.png"
class="wikilink" alt="Pastedimage20260502144709.png" />
<figcaption
aria-hidden="true">Pastedimage20260502144709.png</figcaption>
</figure>

## 使用grasp\_ --all

    python scripts/generate_grasps.py --all

确保你处于 `grasp_generation` 根目录下，直接在终端敲：

**1. 一键可视化所有物体的最佳抓取（每个生成 1 个 HTML）：**

    python tests/visualize_result_all.py --all

**2. 一键可视化所有物体的前 3 种不同抓取姿势：**

    python tests/visualize_result_all.py --all --num_vis 3

可视化 .html 文件会位于data/l20_graspdata

<img src="README-media/987b0a570e6affc1f359f283f4db725fe0de4fd9.png"
class="wikilink" alt="Pastedimage20260510151517.png" />
<img src="README-media/a9ddf4a89e02280162515e7b12eea2578839cd52.png"
class="wikilink" alt="Pastedimage20260510151538.png" />
<img src="README-media/be54ddaec7b6bc57a7f8233b68e0fd6964446006.png"
class="wikilink" alt="Pastedimage20260510151602.png" /><img src="README-media/8967e514a3664467384090287b70f88dd5f22a2a.png"
class="wikilink" alt="Pastedimage20260510151626.png" /><img src="README-media/0bff623b0d6cd1bd20234402ea1431b59b3c5c0c.png"
class="wikilink" alt="Pastedimage20260510151738.png" />

# 环境搭建

参考[environment.yml](https://github.com/brantleeee/DexGraspNet_LinkerHand_l20/blob/master/environment.yml "environment.yml")、[DexGraspNet_env_report_20260516_193632.log](https://github.com/brantleeee/DexGraspNet_LinkerHand_l20/blob/master/DexGraspNet_env_report_20260516_193632.log)
重点注意python版本、PyTorch版本、GCC版本

# 模型准备

### 官方 ShadowHand 的 MuJoCo 动力学测试

先用官方 DexGraspNet 代码为 ShadowHand 分别生成杯子和积木抓取位姿：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
bash scripts/setup_official_shadowhand.sh
bash scripts/generate_official_shadowhand.sh shadow_cup_jenga_006_20261102 16 6000
```

首次运行会从 [PKU-EPIC/DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet)
克隆并固定到提交 `bd1b13d7248729af117e1d46aaa6266b147a3c7b`。
生成在独立的兼容工作树中进行；补丁见
`assets/shadowhand_official/generation_compat.patch`，它使原生成入口支持统一
`scale 0.06`、同时指定两个物体，并兼容本机 PyTorch/trimesh。原方法的
0.2 m 凸包初始化扩张仍保留。积木网格来自本仓库
`data/meshdata/2_of_Jenga_Classic_Game`，不是官方 DexGraspNet 数据集中的积木。
输出分别位于
`external/DexGraspNet_official_compat/data/experiments/shadow_cup_jenga_006_20261102/results/`
下的杯子与积木 `.npy`，每个文件含 16 个原生 ShadowHand 关节抓取候选。
运行名称必须使用未存在的目录；脚本后两个参数分别是每个物体的候选数和优化步数。
生成的静态候选仍需动力学筛选，不能直接视为稳定抓取。

用 MuJoCo 对生成结果做初筛，例如静态能量最低的杯子候选 4、积木候选 7：

```bash
/home/lee/robot_learning/mjlab/.venv/bin/python scripts/test_shadowhand_mujoco.py \
  --backend cpu --scale 0.06 \
  --objects core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03 \
  --result-file external/DexGraspNet_official_compat/data/experiments/shadow_cup_jenga_006_20261102/results/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03.npy \
  --jenga-mode native \
  --jenga-result-file external/DexGraspNet_official_compat/data/experiments/shadow_cup_jenga_006_20261102/results/2_of_Jenga_Classic_Game.npy \
  --indices 4 --jenga-indices 7 --cases gz- lift \
  --output data/experiments/shadow_cup_jenga_006_mujoco_screen
```

本次实际运行的两个候选在 `gz-` 和 `lift` 均未通过严格动力学检查（0/4），
报告位于 `data/experiments/shadow_cup_jenga_006_mujoco_screen/REPORT.md`。
这是两个候选的初筛结果；全部候选的验证结果见下文。

全部 32 个生成位姿的原生 MuJoCo 验证命令（无需 mjlab）：

```bash
.venv_l20/bin/python scripts/test_shadowhand_mujoco.py \
  --backend cpu --scale 0.06 \
  --objects core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03 \
  --result-file external/DexGraspNet_official_compat/data/experiments/shadow_cup_jenga_006_20261102/results/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03.npy \
  --jenga-mode native \
  --jenga-result-file external/DexGraspNet_official_compat/data/experiments/shadow_cup_jenga_006_20261102/results/2_of_Jenga_Classic_Game.npy \
  --output data/experiments/shadow_cup_jenga_006_all_mujoco_20261102
```

已实际完成 16 个杯子和 16 个积木候选的六方向重力加抬升，共 224 场景；
严格通过数为 0/224。全部场景的初始肌腱限位误差都超过 0.1 mm，
其中杯子为 0.9–8.6 mm、积木为 3.0–7.5 mm；多数场景的最大物体接触
穿透也超过 1 mm。完整指标和逐场景轨迹见
`data/experiments/shadow_cup_jenga_006_all_mujoco_20261102/`。

### 官方 Isaac Gym 验证（已在本机运行）

本机已从 NVIDIA 官方下载 Isaac Gym Preview 4，并创建独立 Python 3.8 环境
`.venv_isaacgym`。启动脚本设置 SDK 和 libpython 路径，不依赖 mjlab。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
bash scripts/run_shadowhand_isaacgym.sh \
  --sim-steps 100 \
  --output "data/experiments/shadow_isaacgym_$(date +%Y%m%d_%H%M%S)"
```

直接调用固定版本官方 `IsaacValidator`，默认跑杯子、积木全部 32 个候选，
每个候选六个方向，同时比较官方接触调整与原始位姿。官方接触调整由
`scripts/prepare_shadowhand_isaacgym.py` 在 `.venv_l20` 中计算，已保存到
`data/experiments/shadow_cup_jenga_isaacgym_inputs_20261102/`。
更换生成输入后，应先以该脚本指定一个新 `--output` 目录重新准备，
再向启动脚本传入对应的 `--inputs`。

实际全量结果：官方接触调整后杯子 valid 10/16、积木 6/16；
不做接触调整时杯子 5/16、积木 2/16。
详见 `data/experiments/shadow_cup_jenga_isaacgym_all_20261102/ANALYSIS.md`。
官方成功条件是运行 100 步后六方向均仍有手物接触，并且输入 `E_pen < 0.001`，
不包含此前 MuJoCo 协议的滑移、旋转、持续稳定时间和抬升要求，故不能直接比较通过率。

打开 Isaac Gym 原生窗口查看积木候选 7：

```bash
bash scripts/run_shadowhand_isaacgym.sh \
  --objects jenga --variants official --indices 7 --gui --sim-steps 1000 \
  --output "data/experiments/shadow_jenga_gym_gui_$(date +%Y%m%d_%H%M%S)"
```

改成 `--objects mug --indices 0` 可查看杯子。`--variants no-force`
用于查看不做官方接触调整的位姿。

窗口默认暂停，先点击窗口获取焦点，按 **空格** 开始或暂停。
鼠标左键拖动或方向键可绕物体旋转，PageUp/PageDown 缩放，Home 复位，
**1–6** 切换的是六个重力场景。当前默认验证 1000 步（实际仿真约 16.67 秒），
完成后窗口继续保持打开，关闭窗口退出。默认 `--display-speed 1` 按仿真时间播放；
`--display-speed 0.5` 以半速播放约 33 秒，但物理仿真时长仍为 16.67 秒。
`--sim-steps 100` 可恢复官方原始筛选时长；报告会记录实际步数。
终端每 100 步输出引擎时间，`gui_progress.json` 保存实时步数、状态和时间，
`summary.json` 与 `REPORT.md` 记录引擎实测时长；1000 步实测为 16.666668 秒。
如需自动开始并在完成后 3 秒退出，追加 `--autostart --viewer-seconds 3`。
输出目录保存 `gym_initial.png` 和 `gym_final.png` 两张实际窗口截图。
在本机双显卡环境，启动脚本会为 `--gui` 显式选择 NVIDIA Vulkan ICD，
避免出现 GPU 物理正常但窗口全黑的问题；相机对准所选环境的局部原点。

官方代码的最终验收为：六个方向结束时各至少存在一个手物接触，且输入候选保存的
`E_pen < 0.001`。`E_pen` 是生成阶段物体表面采样点落入手内部的穿透深度之和，
并非 Gym 仿真接触的最大穿透深度，也没有在接触调整后重新计算。
成功条件不含法向力下限、滑移或转角上限、连续保持时间、抬升要求。
1000 步是本地延长测试，官方原始验证类默认 100 步；`official` 仅表示采用官方接触调整。

### MuJoCo 使用 Gym 同类验收协议

新增独立入口 `scripts/validate_shadowhand_mujoco_gym_protocol.py`，原生 MuJoCo，
使用 Gym 同一份接触调整后的位姿、手部 XML、实测质量惯性、22 个位置驱动、
六方向重力和最终接触判据。详细的参数映射和引擎差异见
[对照说明](assets/shadowhand_official/MUJOCO_GYM_PROTOCOL.md)。

重新仿真积木候选 7（1000 个控制步，总物理时间约 16.67 秒）：

```bash
.venv_l20/bin/python scripts/validate_shadowhand_mujoco_gym_protocol.py \
  --objects jenga --variants official --indices 7 --sim-steps 1000 \
  --output "data/experiments/jenga_mujoco_gym_protocol_$(date +%Y%m%d_%H%M%S)"
```

可加 `--gui --autostart` 观察计算过程。为了稳定积分零阻尼、刚度 1000 的驱动，
默认每个控制步使用 512 个数值子步，因此 CPU 计算可能慢于真实时间。
验证后推荐用以下命令在原生 MuJoCo 中按真实时间回放全部方向：

```bash
.venv_l20/bin/python scripts/replay_shadowhand_mujoco.py \
  --protocol gym \
  --results data/experiments/shadow_mujoco_gym_native_all_1000_20261102 \
  --object jenga --variant official --candidate 7 --case gy- --speed 1
```

改成 `--object mug --candidate 0` 可查看杯子。
空格/F8 暂停，1–6 切换六个重力场景，鼠标旋转、平移和缩放。
F5/F6 分别切换双方碰撞几何，F7 切换接触显示。
重新仿真时删除 `--objects ... --indices ...` 可测试两个物体的全部 32 个候选。
本次已实际完成全部 32 个候选、192 个方向场景，均完成约 16.67 秒且没有数值失败，
但杯子、积木均为 0/16 通过六方向接触验收。报告保存在
`data/experiments/shadow_mujoco_gym_native_all_1000_20261102/`。
同为 1000 步的 Gym 基线为杯子 valid 10/16、积木 6/16；
接触预测与数值积分仍有引擎差异，详见该目录 `ANALYSIS.md`。

### ShadowHand MuJoCo 结果可视化

批量导出全部 32 个候选的抬升轨迹视频和杯子、积木各一张初始位姿总览图：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
.venv_l20/bin/python scripts/render_shadowhand_batch.py
```

输出在 `data/experiments/shadow_cup_jenga_006_all_mujoco_20261102/visualizations/lift/`。
其中 `*_contact_sheet.png` 是 16 个候选的总览，`*.mp4` 是各候选的
实际 MuJoCo 动力学轨迹回放。可将 `--case lift` 改为 `gz-` 或其他重力方向。
加 `--show-contacts` 会另存至 `visualizations/lift_contacts/`，显示接触点
和接触力；严重穿透时力箭头可能遮挡模型。

在有桌面显示的终端打开原生 MuJoCo 交互窗口，例如杯子候选 4 与积木候选 7：

```bash
.venv_l20/bin/python scripts/replay_shadowhand_mujoco.py \
  --object mug --candidate 4 --case lift
.venv_l20/bin/python scripts/replay_shadowhand_mujoco.py \
  --object jenga --candidate 7 --case lift
```

`--candidate` 可取 0–15，`--case` 可取 `gx+`、`gx-`、`gy+`、`gy-`、
`gz+`、`gz-`、`lift`。F5 切换手部碰撞体，F6 切换物体碰撞网格，
F7 切换接触显示，F8 暂停；暂停时左右方向键逐帧查看。ShadowHand 手部
碰撞体是盒体和胶囊体，物体碰撞几何为凸网格；两者可分别显示。

新增独立入口 `scripts/test_shadowhand_mujoco.py`，直接读取固定版本的官方
ShadowHand 抓取及物体网格，不把 L20 结果映射到 ShadowHand。支持 CPU MuJoCo
和 CUDA mjlab，逐一测试六方向重力与抬升，保存 XML、轨迹和完整报告。

```bash
/home/lee/robot_learning/mjlab/.venv/bin/python scripts/test_shadowhand_mujoco.py \
  --backend cpu --scale 0.06 --limit 2 \
  --output data/experiments/shadowhand_manual
```

`--limit 2` 表示每个物体选两个候选；删除 `--limit 2 --scale 0.06` 可测试全部
1,141 个官方样例及其原始尺度。输出目录必须是新目录。本入口保留官方原始位姿，
不执行局部优化；仿真完成与抓取通过分别统计。模型兼容修复、机械参数及授权来源见
[说明](assets/shadowhand_official/ADAPTATION.md)。

官方 ShadowHand 杯子与本地积木的对照命令：

```bash
/home/lee/robot_learning/mjlab/.venv/bin/python scripts/test_shadowhand_mujoco.py \
  --backend cpu --objects core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03 \
  --jenga-mode transfer --indices 0 --scale 0.06 \
  --output "data/experiments/shadowhand_cup_jenga_$(date +%Y%m%d_%H%M%S)"
```

杯子使用固定版本官方 ShadowHand 位姿和物体网格；积木使用同一官方 ShadowHand
位姿、仓库本地积木网格。**积木是跨物体迁移诊断，并非官方积木抓取位姿**。
每个物体均测试六方向重力和抬升，报告记录位姿来源及物体网格 SHA256。
如已另有 ShadowHand 关节命名的积木 `.npy`，可改用
`--jenga-mode native --jenga-result-file <路径>`；L20 位姿会因关节名不匹配被拒绝。

2026-11-02 已对官方杯子 scale 0.06 候选 0 与其积木迁移姿态完成 CPU 七场景验证：
两者各 0/7 通过；CUDA mjlab 后端的 `gz-`、`lift` 两场景也各 0/2 通过。
这说明**当前输入位姿在该 MuJoCo 协议下失败**，不能推论官方 ShadowHand
对积木本身不可抓。杯子候选 0 在当前 ShadowHand 模型中初始肌腱限位误差约
1.86 mm，且两物体测试最终均失去接触；这是解释结果时的重要协议差异。
完整数据分别见 `data/experiments/shadowhand_mug_jenga_transfer_all_cpu_20261102/`
与 `data/experiments/shadowhand_mug_jenga_transfer_gpu_20261102/`。

### shadow hand模型可视化

<figure>
<img src="README-media/20ddfaea22efc00606c72cb4bb53b49745640456.png"
class="wikilink" alt="Pastedimage20260509230521.png" />
<figcaption
aria-hidden="true">Pastedimage20260509230521.png</figcaption>
</figure>

经过可视化校准，迁移有明显需要注意的

ShadowHand body frame == LinkerHand body frame
ShadowHand geom frame == LinkerHand geom frame
但是 ==mesh 零姿态语义不同==
LinkerHand L20 需要绕自身 Z 轴 -90° 才和 ShadowHand 掌心/四指方向一致
现在出现的是"物理外观对齐"和"坐标轴对齐"二选一的现象。这不是脚本错，而是说明 LinkerHand 的 mesh 语义帧和 body frame 语义帧不是 ShadowHand 那套定义。

<img src="README-media/a05e1f18c5d8cea224b258daa1783ca67c5b24b2.png"
class="wikilink" alt="Pastedimage20260502204605.png" />
现在出现的是"物理外观对齐"和"坐标轴对齐"二选一的现象。这不是脚本错，而是说明 LinkerHand 的 mesh 语义帧和 body frame 语义帧不是 ShadowHand 那套定义。
\## 接触点和穿透点生成


    cd /root/DexGraspNet/grasp_generation

    python generate_l20_contacts_right_v1.py

    python generate_l20_armor_right_v2.py

    python check_contact_fk_l20_right.py

    python check_penetration_fk_l20_right.py

<img src="README-media/79a4423744ce3610150cf501114c348963d5d8f9.png"
class="wikilink" alt="Pastedimage20260509200719.png" />
<img src="README-media/4245b8b72f72782655a8736c0b13af759b88ffe4.png"
class="wikilink" alt="Pastedimage20260509202221.png" />
<img src="README-media/afa2184ee3a563872f352c663c7819143326dd1e.png"
class="wikilink" alt="Pastedimage20260509202101.png" />
原shadowhand

    穿透点
    ✅ 统计完成！
    -> 包含点云的连杆总数 : 23 个
    -> 全局暴露面防穿模点总计: 41 个
    =================================================

    接触点
    ✅ 统计完成！
    -> 包含点云的连杆总数 : 19 个
    -> 全局暴露面防穿模点总计: 140 个
    =================================================

### 对比

#### shadow hand

<figure>
<img src="README-media/20ddfaea22efc00606c72cb4bb53b49745640456.png"
class="wikilink" alt="Pastedimage20260509230521.png" />
<figcaption
aria-hidden="true">Pastedimage20260509230521.png</figcaption>
</figure>

#### 修改后的linkerhand l20

<figure>
<img src="README-media/c685c75f3d377d403be1770f2602c806828a6d45.png"
class="wikilink" alt="Pastedimage20260510120119.png" />
<figcaption
aria-hidden="true">Pastedimage20260510120119.png</figcaption>
</figure>

## 📚 Citation

This project is built upon the [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) framework.  
If you use this code or dataset in your research, please cite the original work:

```bibtex
@article{wang2022dexgraspnet,
  title={DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation},
  author={Wang, Ruicheng and Zhang, Jialiang and Chen, Jiayi and Xu, Yinzhen and Li, Puhao and Liu, Tengyu and Wang, He},
  journal={arXiv preprint arXiv:2210.02697},
  year={2022}
}

```

If you additionally use the modifications or extensions provided in this repository (e.g., LinkerHand L20 adaptation), please cite as:
```bibtex
@misc{your2026dexgraspnetl20,
  author = {Li Juyang},
  title = {DexGraspNet LinkerHand L20 Extension},
  year = {2026},
  howpublished = {\url{https://github.com/brantleeee/DexGraspNet_LinkerHand_l20}},
  note = {GitHub repository}
}
```


### mjlab 残差策略与稳定性修正

L20 已新增独立的 [Isaac Sim SDF 动力学验证](docs/ISAACSIM_L20_SDF.md)：完整手部 STL 使用 SDF 碰撞，可验证杯子、积木的保存抓取位姿，支持六方向测试、GUI 和批量筛选。文档同时说明现有物体网格的来源限制。

静态参考使用 scale `0.06`、组内候选 `7`。旧 v1 的 10000 轮训练仍未满足严格稳定抓取条件，不能仅凭总奖励判断成功。实验 v2 增加旋转和速度约束、关节目标限速及更长的抬升后保持时间，但 200 轮试训尚未通过稳定性评估，不应直接长训。旧模型查看时保留其原任务版本；当前新训练使用下述 v3。版本说明、训练及逐回合评估命令见 [mjlab_grasp/README.md](mjlab_grasp/README.md)。

v3 使用 **稳定保持奖励**：接触留住杯子、朝向、抬升及低速度保持分项奖励，掉落一次性扣 25 分；保留 v1 的关节控制和 scale 0.06 候选 7 零位。可从原策略仅复用 actor，重训价值网络。训练、续训及评估说明见 [v3 奖励说明](mjlab_grasp/README.md#v3按稳定保持重新设计奖励)。奖励修改后的模型仍须通过连续稳定保持评估。

当前新运行默认 **v4**。实验性的 v5 在抬升后限制关节目标变化并加入连续收敛奖励，但 100 轮短训后掉落显著增多，暂不建议长训。诊断、现成 v4 模型的限速对照及后续物理筛选建议见 [v5 说明](mjlab_grasp/README.md#v5抑制掉落时的手指目标振荡)。
