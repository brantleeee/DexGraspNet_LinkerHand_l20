# region_exports：区域约束DexGraspNet＋ShadowHand

2026-11-15。源目录：`/home/lee/robot_learning/pose_estimation/rm_arm_mujoco/assets/objects/power_drill/grasp_clean/region_exports`。

## 输入与规则

本目录不是五个独立物体。manifest将690面标为SAFE_GRASP、3851面为CONDITIONAL、98面AVOID_CONTACT、3046面FORBIDDEN、3579面UNKNOWN。默认接触仅允许类别0；未授权类别1作为接触目标，类别2/3/4同样排除。

先验证manifest源OBJ SHA、semantic_surface的顶点/面顺序与源一致、各区域face_ids和标签逐项一致。源SHA为47d5fba7868be887e7b58003bc2ccd9372a89e173c970ccb25039af0e62b4d54。保存完整region_source副本和generation_surface.obj，物体坐标及尺寸保持，scale=1，255.5×223.4×75mm。

完整表面未封闭，因此为完整物体距离查询另建非凸实体并集：补80个缺失孔壁面，不用凸包或减面，不重新居中。原区域面标签始终针对原始表面；距离代理新面不假装继承语义标签。用完整代理采样2000物体表面点计算原DexGraspNet手内穿透项，不把safe_grasp.obj当独立可抓物体。

## 算法及与原方法的区别

`scripts/dexgraspnet_drill_regions.py`复用官方兼容ShadowHand模型、22关节、接触候选、原始E_fc/E_pen/E_spen及Annealing优化器。

- 接触吸引项改为手选接触点到SAFE_GRASP三角面的无符号距离和，法向/原力闭合计算仍来自完整闭合物体。
- 对手部128个采样表面点与所有非SAFE三角面计算距离，惩罚小于2mm的距离。额外权重默认100，是明确的软惩罚，并不保证连续全手间隙、无穿透或闭合路径安全。
- 初始化在SAFE面按面积采样，用完整实体的外法向设置接近姿态，腕部距离30–70mm；没有复用原论文200mm凸包外扩，因此不是原论文初始化的严格复现。
- 每次提议投影到原关节范围，再重算完整运动学；断开跨优化迭代的计算图，保留当前步梯度。不会事后悄悄截断导出角度。
- 默认32候选、6000步、seed42、每手4个选中接触点，输出energy包含新增语义项，不能与旧原始能量直接排名。

输出每候选保存E_fc/E_dis/E_pen/E_spen/E_joints/E_semantic、选中接触最近类别、到SAFE最大距离、非SAFE手部采样点最小距离。`semantic_verified=false`及`physics_validated=false`始终保留；这些采样值不是严格碰撞验收。

## 当前执行

短检查dgn_drill_regions_smoke_v3_20261115两候选×20步完成：梯度/数值/导出正常，0候选全部接触最近SAFE、0候选全部接触距离小于1mm，不能当抓取生成已成功。2项面标签/源文件身份测试通过，语法检查通过。

正式任务已启动：`data/experiments/dgn_shadow_drill_regions_20261115`。只做生成，不启动Isaac Sim或训练。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
tail -n 12 data/experiments/dgn_shadow_drill_regions_20261115.log

# 等生成和导出完成
xdg-open data/experiments/dgn_shadow_drill_regions_20261115/index.html
cat data/experiments/dgn_shadow_drill_regions_20261115/summary.json
```

生成结果：results/power-drill.npy（原下标32候选）、states.npz（初始/最终手姿态、接触点/手表面样本、区域距离）、summary.json及前16彩色HTML。绿色SAFE、蓝色CONDITIONAL、黄色AVOID、红色FORBIDDEN、灰色UNKNOWN；手部淡蓝色。HTML均为静态姿态，不是动力学回放。

以后新批次命令（当前任务运行中，不要重复启动）：

```bash
region_out="data/experiments/dgn_drill_regions_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/dexgraspnet_drill_regions.py \
  --regions /home/lee/robot_learning/pose_estimation/rm_arm_mujoco/assets/objects/power_drill/grasp_clean/region_exports \
  --batch-size 32 --n-iter 6000 --top-k 16 --output "$region_out" \
  > "$region_out.log" 2>&1
```

后续先检查接触是否真正落在SAFE、非SAFE间隙、全手三角形相交及闭合轨迹，再进行物理筛选。结果不合格时优先分析接触分配与语义几何，不把优化结束或低能量视为稳定抓取。
