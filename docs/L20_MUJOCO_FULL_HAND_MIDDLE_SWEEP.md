# MuJoCo全手中指完整弯曲自碰撞检查（2026-11-17）

输出：`data/experiments/l20_middle_mujoco_full_hand_20261117`。已实际完成，不是待运行命令。

**结论：在本次4212个采样构型中，0.01mm阈值以上的中指新增接触仅为middle_metacarpals/middle_proximal根部相邻结构，最大原生SDF深度0.07677mm。未检出中指与掌体、其他手指的超阈值碰撞。整手起点存在拇指装配接触，故不能表述成“整手没有自碰撞”。**

## 模型与覆盖范围

- MuJoCo 3.17.0，原L20 URDF的22个link/22个碰撞STL，全部保留；22个原生mesh SDF，不使用凸包、凸分解、简化碰撞网格或接触点替代碰撞体。
- 显式列出22选2=231组link配对，包括父子连杆和固定连接。关闭自动碰撞掩码，使用显式pair控制，避免默认父子/固定体过滤漏检；配对集合逐项与所有body组合核对相同。
- 5个mimic等式保留，逐帧还直接设置符合关系的DIP；中指DIP=.8917×PIP，记录最大误差0。
- MCP_pitch 0→1.4rad，PIP 0→1.57rad，DIP 0→1.399969rad。其他中指关节按背景固定。
- 五条路径：单独MCP；PIP/DIP联动；同步关闭；先MCP后PIP；先PIP后MCP。
- 四种背景：其余关节零位；食指/无名指/小指完全闭合而拇指仍零位；原候选21的其他关节；原候选7的其他关节。候选背景保留中指MCP_roll，其余路径根部roll为0。
- 每关节相邻帧增量≤0.01rad（约0.573°），20组路径，共4212帧。所有link对都在每帧检查，不只检查中指与某几个目标。

测试没有电钻、重力、平面或主动电机控制。直接指定qpos并调用`mj_forward`，用于检测整个弯曲构型序列，包括假如受阻后继续弯曲会出现的碰撞；不是PD动态执行，也不是连续碰撞检测CCD或整个二维关节空间认证。

## 实际检出

|情况|结果|
|---|---|
|中指—掌体/其他手指|本次路径中未检出深度>0.01mm的接触|
|中指掌骨—近节|MCP弯曲路径有新增小接触，最大0.07540–0.07677mm|
|仅PIP/DIP弯曲|四背景均无新增>0.01mm接触对|
|其他link的原有接触|全部记录，未作为中指新碰撞；零位背景最大约3.98107mm，位于拇指掌骨与基座1|

同步路径首次越过阈值约在MCP=.00892/PIP=.01000rad，MCP单独路径约在MCP=.02rad。它位于同一铰接根部，并非中指碰到食指/无名指。但不能仅凭相邻关系就宣布可忽略：需要分清原CAD装配/原生SDF数值误差与真实机械干涉。

表格里的整手最大深度包含拇指已有装配接触，不是中指弯曲造成的最大值。逐帧JSON保留阈值以下接触及位置/法向；报告中的“新增”只是相对同背景同路径起点的配对集合，不丢弃基线。

## 几何与实现核对

- 原始STL/URDF SHA未变；22link在伸直和全弯两个端点的世界网格包围盒与生成器FK核对通过，最大差5.005e-7m（约0.5微米）。
- 231个显式配对集合完整、4212帧角步长/轨迹尺寸、mimic和仅一组中指新增对核对通过。
- 2项测试通过：五条路径端点/最大角步长；即便自动掩码为0且同一固定体，显式SDF pair仍实际检出重叠。
- 13个源网格封闭、9个不封闭；原生SDF是数值近似，报告深度不是独立的原三角面精确穿透测量。不能据未检出就认证所有姿态或连续路径无实体碰撞。
- 伸直到全弯视频由实际MuJoCo Renderer离屏渲染，640×480、30fps、181帧、约6.03秒；这是展示时长，不是物理执行用时。已抽帧核查完整手部可见。修复了默认第3碰撞组隐藏和离屏帧缓冲尺寸限制，未改变扫描结果。
- `sweep_source_at_run.py`保存扫描时源码及原SHA；当前入口另有仅回放显示修正。

## 视频与交互回放

[MuJoCo原生整手弯曲视频](../data/experiments/l20_middle_mujoco_full_hand_20261117/neutral_coupled_mujoco.mp4)

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/experiments/l20_middle_mujoco_full_hand_20261117/index.html

# 原生MuJoCo窗口，循环回放，鼠标可转动视角；关闭窗口结束。
.venv_l20/bin/python scripts/check_middle_mujoco_full_hand.py \
  --output data/experiments/l20_middle_mujoco_full_hand_20261117 \
  --replay --case neutral__coupled
```

蓝色为普通link、橙色为中指link；某link任一接触深度>0.01mm时整link着红。红色不代表该link整个体积都穿透，也包含已有拇指装配接触。可切换`--case candidate_21__mcp_then_pip`、`candidate_7__pip_then_mcp`或其他已保存名称。

## 重新执行

```bash
l20_sweep="data/experiments/l20_middle_full_hand_$(date +%Y%m%d_%H%M%S)"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv_l20/bin/python \
  scripts/check_middle_mujoco_full_hand.py \
  --max-step-rad 0.01 --output "$l20_sweep"

# 用同一原生MuJoCo模型导出视频，不重新运行动力学。
MUJOCO_GL=egl .venv_l20/bin/python scripts/check_middle_mujoco_full_hand.py \
  --output "$l20_sweep" --case neutral__coupled \
  --video "$l20_sweep/neutral_coupled_mujoco.mp4"
```

## 对抓取生成问题的意义

此前自排斥中最强的middle_proximal/index_proximal、middle_proximal/ring_proximal，在本次所测路径没有出现超阈值SDF接触。这支持继续检查20mm关键点代理过于保守，而不是直接认定中指因为撞上其他手指而不能弯曲。

接触点/法向库与碰撞网格是两层数据；v6改了优化候选与方向，没有改变手部实体STL。下一步可做分组接触目标和真实间隙排斥对照，同时保留全手碰撞护栏及根部小接触记录。当前没有重跑抓取生成或验证电钻承载。
