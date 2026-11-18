# 食指、无名指、小指、拇指：MuJoCo整手弯曲碰撞检查

日期：2026-11-17。输出：`data/experiments/l20_other_fingers_mujoco_20261117`。

沿用中指实验的完整L20右手模型：22个原URDF碰撞link、22个原生mesh SDF、全部231组显式link配对，含父子及固定连接。未使用凸包或简化碰撞网格，也没有用接触候选点代替实体碰撞体。

## 本次结果（已完成）

44条路径、9288帧，约245秒扫描，mimic误差为0。配对数、角步长、路径长度与3项测试核对通过。

|手指|路径数|本次主要新增接触|
|---|---:|---|
|食指|10|掌骨/近节相邻根部，最大约0.08049mm|
|无名指|10|掌骨/近节相邻根部，最大约0.07647mm|
|小指|10|掌骨/近节相邻根部，最大约0.07642mm|
|拇指|14|零位背景部分路径新增掌体/拇指远节，最大约0.37938mm；其他普通手指闭合背景共有25种新增连杆对，新增对最大约1.79356mm|

拇指已有装配接触的全程最大值约3.98107mm，不能与“新增最大1.79356mm”混为同一指标。结果支持进一步关注拇指与掌体/其他手指的干涉，不能宣称所有手指弯曲均无碰撞。完整首次关节角见自动REPORT和contacts.json。

已导出5个原生MuJoCo视频（四指零位背景同步运动，以及拇指其他普通手指闭合背景），均640×480、30fps、约6.03秒；已读取核对，并抽帧确认完整手部及活动手指显示。统一入口：`data/experiments/l20_other_fingers_mujoco_20261117/index.html`。

## 路径与背景

- 食指、无名指、小指：MCP 0→1.4rad；PIP 0→1.57rad，DIP=.8917PIP。每指单独MCP、单独PIP/DIP、同步、先近端后远端及反序，共5路径。
- 拇指：CMC yaw 0→1.4、roll 0→1.22、pitch 0→.79、MCP 0→1.05rad；DIP=1.1619MCP。四独立关节分别扫描，再同步、先近端、先远端，共7路径。独立关节极限同时组合是URDF允许的探查构型，不代表硬件控制一定可达。
- 每根手指分别测试`neutral`和`others_closed`背景。后者把其他普通四指的MCP/PIP关闭，拇指仍零位（当拇指为测试对象时按其路径运动）。这两种背景不是对其余手指所有可能姿态的穷举。
- 角步长≤.01rad，深度阈值.01mm，保留起点装配接触、全部逐帧接触及新增接触。记录的是原生SDF深度估计，开放STL符号仍有限制。

这是运动学逐帧设置qpos并调用mj_forward的自碰撞检查，没有物体、重力、力矩控制；有限角采样不是连续CCD或动力学验证。报告中的“新增”只表示该连杆对在路径起点未超阈值，不表示起点原有碰撞可以忽略。

## MuJoCo交互查看

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

# 食指：完整手部始终显示，只移动食指。
.venv_l20/bin/python scripts/check_fingers_mujoco_full_hand.py \
  --output data/experiments/l20_other_fingers_mujoco_20261117 \
  --replay --case index__neutral__coupled
```

将case分别换为：

|手指|同步弯曲case|
|---|---|
|食指|index__neutral__coupled|
|无名指|ring__neutral__coupled|
|小指|pinky__neutral__coupled|
|拇指|thumb__neutral__coupled|

把neutral换成others_closed可看其他普通手指闭合背景。鼠标转动/缩放；空格暂停/继续，左右方向键逐帧查看。关闭窗口结束，窗口不启动新扫描。

蓝色为其他link、橙色为当前测试手指；任一接触超过.01mm时整link着红，包含起点已有装配接触。不能把整link红色解释为整个link都穿透。所有显示组已启用，避免只看到接触标记而看不到完整模型。

```bash
# 依次打开四根手指，关闭当前窗口才会打开下一根。
for finger in index ring pinky thumb; do
  .venv_l20/bin/python scripts/check_fingers_mujoco_full_hand.py \
    --output data/experiments/l20_other_fingers_mujoco_20261117 \
    --replay --case "${finger}__neutral__coupled"
done
```

## 重新执行及导出

```bash
l20_other="data/experiments/l20_other_fingers_$(date +%Y%m%d_%H%M%S)"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv_l20/bin/python \
  scripts/check_fingers_mujoco_full_hand.py --output "$l20_other"

# 原生MuJoCo离屏渲染，输入为已保存轨迹。
for finger in index ring pinky thumb; do
  MUJOCO_GL=egl .venv_l20/bin/python scripts/check_fingers_mujoco_full_hand.py \
    --output "$l20_other" --case "${finger}__neutral__coupled" \
    --video "$l20_other/${finger}_mujoco.mp4"
done
```

原始结果见输出目录的REPORT.md、results.json、各路径contacts.json/trajectory.npz及collision_audit.json。3项测试通过，核对所有手指路径的角范围/步长/mimic，以及显式原生SDF配对确实覆盖固定体。

观察时先区分：起点已有装配交叠、弯曲中新出现的相邻关节接触、手指对掌体/其他手指的接触。不能仅凭SDF小接触就判断真实机械干涉，也不能据某条轨迹未检出就认证整个关节空间无碰撞。
