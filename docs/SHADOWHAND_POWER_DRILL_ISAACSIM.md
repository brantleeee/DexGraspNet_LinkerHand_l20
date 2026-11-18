# ShadowHand 电钻生成位姿的 Isaac Sim SDF 验证

入口：`scripts/validate_shadowhand_isaacsim.py`。这是静态生成位姿的物理保持验证，无RRL，无官方Gym接触调整，也不进行张开闭合。

默认按保存能量选择前16个候选，每候选每方向一个独立Isaac Sim进程，固定手腕和原关节目标，立即施加六个模型初始坐标轴方向的重力，保持10秒。物体自由运动，无支撑、无固定物体约束。即使失败也记录全程轨迹，不是MuJoCo的首次失败提前结束版本。

## 使用

先等待生成目录中出现完整`results/power-drill.npy`，不要读取生成中的未完成文件。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
shadow_result="external/DexGraspNet_official_compat/data/experiments/shadow_power_drill_20261114_233746/results/power-drill.npy"
shadow_check="data/experiments/shadow_drill_sdf_$(date +%Y%m%d_%H%M%S)"

# 可选：先只导出输入、核对每个候选的正运动学；不会启动Isaac Sim。
.venv_l20/bin/python scripts/validate_shadowhand_isaacsim.py \
  --result-file "$shadow_result" --top-k 16 \
  --prepare-only --output "${shadow_check}_prepare"

# 正式验证：16×6方向，默认10秒重力保持。
.venv_l20/bin/python scripts/validate_shadowhand_isaacsim.py \
  --result-file "$shadow_result" --top-k 16 \
  --gravity-seconds 10 --sdf-resolution 256 \
  --output "$shadow_check"

xdg-open "$shadow_check/index.html"
```

`--indices 0 --cases gy+ --gui`可选单候选单方向显示；不加`--indices`时按能量排序，`--indices`始终是NPY原始下标。输出必须为新目录；本版未实现续跑，失败日志和已完成结果保留，不能原目录重复执行。

## 模型与控制

- 读取官方兼容工程实际HandModel生成的几何：掌部box、指节capsule/mesh等已有简化几何的三角化表面；不是把可视化外壳换成真实精细手部。所有非空手部几何与电钻使用SDF，无新凸包/凸分解。
- 保持生成库内部joint/link变换，吸收固定预关节帧；手部几何、质心和惯量随坐标变换同步转换。每个选中候选与实际生成库FK比较，矩阵最大误差大于2e-6拒绝导出。
- 22个独立手指关节、23个刚体，不引入L20的mimic或额外ShadowHand tendon耦合。原生成器简化模型不等同于完整商品手硬件。
- Kp=5Nm/rad、Kd=0.05Nm·s/rad、单关节力矩上限1Nm、摩擦3，时间步1/240秒；与官方Gym的1000/0不同，不称为官方Gym协议或硬件标定参数。
- 按每候选保存scale读取物体及手腕，绝不二次缩放位姿。源尺寸电钻应保存约0.2555；质量为封闭网格体积×500kg/m³估算。不能把原件单位未知解释为实测米制尺寸。
- 共享现有原生SDF后端及进程锁；其他Isaac Sim任务占锁时排队，不停止它们。旧L20/RRL源代码和检查点不改动。

## 验收与输出

10秒内最大平移偏差≤5mm，最大转角≤15°，最大接触穿透≤1mm，末尾连续1秒法向合力>0.001N，轨迹数值有效。direct模式不设置零重力获取阶段，通过不证明手能从张开状态获得此抓取。

`inputs/manifest.json`记录原始下标、能量、比例、源NPY/网格散列、手部模型、FK误差、质量惯量；`plan.json`记录运行设置。`results.json`区分物理失败与执行异常，`status.json`记录完整性；`index.html`链接每个方向HTML轨迹回放，显示实际手部/物体SDF输入三角网格。回放不重新做物理。各方向目录另存碰撞/运行时驱动审计、NPZ、summary及错误日志。

CPU自动检查使用已有杯子ShadowHand位姿作为接口测试夹具，对测试夹具副本的角度限制在源关节范围后检查16候选FK/排序/比例，并检查混合比例拒绝；真实输入不截断，越界候选直接拒绝；不属于电钻仿真结果。当前实际电钻生成NPY尚未就绪，因此本次未启动完整GPU验证，也未证明任何电钻方向通过。
