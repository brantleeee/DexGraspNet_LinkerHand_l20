# MuJoCo 对照 Gym 接触验收协议

入口为 `scripts/validate_shadowhand_mujoco_gym_protocol.py`，使用原生 C MuJoCo，
没有调用 mjlab。它独立于之前的严格滑移、旋转、穿透和连续保持验证入口。

## 输入与机械参数

- 与 Gym 共用 `shadow_cup_jenga_isaacgym_inputs_20261102/*.npz` 中的
  `force_qpos` 或 `raw_qpos`、手根部变换、物体 scale、原始 E_pen。
- 直接展开 Gym 使用的 `open_ai_assets/hand/shadow_hand.xml`，保留同一套手部
  连杆坐标和碰撞体。仅替换引擎不兼容的设置以及驱动、限位的表示方式。
- `ISAAC_GYM_RUNTIME_MECHANICS.json` 保存本机从 Gym 导入后的实际参数，
  包括每个关节的驱动、每条肌腱，以及手和物体各刚体的质量、质心、惯性。
- 手根部固定，22 个关节均为位置驱动，kp=1000、物理阻尼=0。
  18 个关节采用 Gym 导出的有限力矩上限，其余四个关节按 Gym 的近似无穷上限处理。
- 实际 Gym 关节属性 `hasLimits=false`，因此该对照关闭 MuJoCo 的硬关节限位。
- Gym 四条肌腱的 `stiffness=0`、`limit_stiffness=1`、`damping=0`，
  限位为 ±0.001 m。MuJoCo 以刚度 1、同范围的死区弹簧表示该软限位，
  不再使用此前严格模型的硬肌腱约束。
- 同一组物体凸块按 scale 缩放；Gym 的多个固定链接刚体在 MuJoCo 中合并成一个
  刚体，质量、质心和惯性按刚体合成公式使用 Gym 实测值。
- 摩擦系数 3、世界重力 `(0,-9.8,0)`，手物碰撞开启、手部自碰撞关闭。

## 时间与数值积分

默认 1000 个控制步，每个控制步为 1/60 s，总物理时间为 16.666667 s，
恰为官方 100 步的 10 倍。六个方向同步推进。

默认 `--drive-integration native --substeps 512`：保持原生 MuJoCo 位置驱动
kp=1000、kv=0，每个控制步细分为 512 个数值积分步（h≈32.55 µs）。
PhysX 的隐式驱动可以使用两个子步，但原生 MuJoCo 零阻尼位置驱动直接采用
该大步长会数值发散；更少的 128 个子步也在长时参考测试中发散。因此，数值积分
子步数是明确保留的引擎差异，不应把两者表述为完全相同的求解模型。

`--drive-integration implicit --substeps 2` 是更快的数值近似：用 kv=h*kp
补偿刚度。在孤立线性系统中对应位置弹簧的后向欧拉离散，但在力矩饱和和接触下
不与 PhysX 严格等价。该模式的结果不能代替上述原生零阻尼参考结果。

积分过程检查 MuJoCo 数值警告、有限状态和引擎时间。出现发散或时间重置时，
记录为数值失败，并强制禁止通过接触验收。

## 最终判据及接触壳

六个独立旋转的方向顺序为 `gy-、gy+、gx-、gx+、gz+、gz-`。
每个方向结束时只要仍有一个手物接触列表项，就记为该方向成功；
候选须六方向全部成功且输入保存的 E_pen < 0.001。
不加入之前严格协议的滑移、转角、持续保持、力下限或抬升要求。

PhysX 的 contact_offset 与 MuJoCo 的 margin/gap 没有一一对应关系。
本实现默认 `--contact-shell-mode final --contact-shell 0.01`：
按真实表面进行物理积分，最终查询时以 margin=gap=10 mm 建立无预接触力的
邻近接触壳，近似 Gym 的接触列表判据。报告同时记录接触列表数量、可产生约束力的
接触数和位移，避免把正间隙的候选接触误读为实际承载接触。

可用 `--contact-shell-mode continuous` 在积分全过程保留 margin=gap 接触壳。
这会改变碰撞检测与接触流形，也不等价于 PhysX 的预测接触。
原生参考报告必须注明所用模式，不能把接触偏置差异归结为模型质量差异。
Gym 的关节速度上限和固定链接柔度也没有在 MuJoCo 中严格复现。

## 运行与可视化

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
.venv_l20/bin/python scripts/validate_shadowhand_mujoco_gym_protocol.py \
  --objects jenga --variants official --indices 7 --sim-steps 1000 \
  --output "data/experiments/jenga_mujoco_gym_protocol_$(date +%Y%m%d_%H%M%S)"
```

加 `--gui --autostart` 可实时观察正在计算的仿真，但 CPU 小步长积分可能比真实时间慢。
完成后可在原生 MuJoCo 窗口按 1 倍速回放已保存的轨迹：

```bash
.venv_l20/bin/python scripts/replay_shadowhand_mujoco.py \
  --protocol gym --results <验证输出目录> \
  --object jenga --variant official --candidate 7 --case gy- --speed 1
```

空格/F8 暂停，1–6 切换重力场景，鼠标旋转、平移和缩放。
F5/F6 分别切换手部和物体碰撞几何，F7 显示接触。
`scene.xml`、`trajectory.npz`、`simulation.json` 和 `results.json` 保存于每个候选目录。
