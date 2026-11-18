# DexGraspNet L20：指定model_clean.obj重新生成

2026-11-15。本轮使用本工程已有L20适配，不使用ShadowHand或固定手指模板。

源：`/home/lee/robot_learning/pose_estimation/rm_arm_mujoco/assets/objects/power_drill/grasp_clean/model_clean.obj`。

新增独立物体资源`data/meshdata/power-drill-clean-l20/coacd/decomposed.obj`，与源逐字节一致，SHA256为47d5fba7868be887e7b58003bc2ccd9372a89e173c970ccb25039af0e62b4d54。旧power-drill资源及所有抓取不改动；此处coacd/decomposed.obj只是加载器历史路径，不表示执行了凸包/凸分解。

## 输入和设置

- scale=1，保持原坐标与255.5×223.4×75mm尺寸，无居中、变轴、再归一化或网格修补。
- 完整OBJ参与生成，不使用region_exports区域约束；不是上一批区域ShadowHand的同条件手型A/B实验。
- L20当前URDF，21个手指关节中16独立优化、5个mimic联动；实际手部网格距离查询force_mesh_sdf开启。
- 32候选，6000步，scale-aware初始化，能量前16静态HTML。初始化会使用物体凸包，但没有把原对象距离网格替换为凸包。
- 原clean面仍不封闭，TorchSDF符号距离对其内外判断可能不可靠，import.json明确记录；不能把文件名clean当作实体合法性证书。
- 静态能量不包含600g实际负载验证，本次没有Isaac Sim或RRL。以后动力学质量仍按用户确认的0.6kg设置，需另指定可靠物理模型。

生成/网格及URDF源散列记录于`data/experiments/l20_clean_drill_20261115.plan.json`。分比例结果manifest中normalized字段是旧命名；本次实际输入未归一化，scale1指原物理数值尺寸。

## 执行状态与查看

短检查`l20_clean_drill_smoke_20261115`两候选×两步已完成：2/2关节/mimic可行，0/2几何合格，2HTML及比例检查正常。不能称已生成稳定抓取。

正式任务`l20_clean_drill_20261115`已启动，不重复启动或持续监控。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
tr '\r' '\n' < data/experiments/l20_clean_drill_20261115.log | tail -n 12

# 生成和导出结束后
xdg-open data/experiments/l20_clean_drill_20261115/results/scales/scale_1/top_16/index.html
cat data/experiments/l20_clean_drill_20261115/scale_validation.json
```

主结果：`results/scales/scale_1/power-drill-clean-l20.npy`。top_16/ranking.json记录能量排名与原下标；排名不等同几何或动力学通过。

## 以后独立批次命令

资源已准备好，以下命令只用于新批次，当前任务无需重跑：

```bash
l20_clean_run="l20_clean_drill_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/generate_small_grasps.py \
  --object-code power-drill-clean-l20 --scales 1 \
  --batch-size-per-scale 32 --n-iter 6000 --export-top-k 16 \
  --hull-inflation scale-aware --name "$l20_clean_run"
```
