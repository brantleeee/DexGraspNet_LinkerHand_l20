# 本地 GraspGenX：电钻＋ShadowHand 固定闭合构型

本地源：`/home/lee/robot_learning/open_source_baselines/GraspGenX`；使用其`.venv`、release/gen/epoch_736.pth及dis/epoch_1056.pth。不修改外部工程、不重新训练、不联网下载。

## 本次接口的真实范围

本地没有官方ShadowHand gripper描述文件。发布模型接受开/半开抓取空间的两个包围盒，生成6自由度刚体位姿和判别分数；不会为每个抓取生成22关节角。

本工程脚本`scripts/graspgenx_shadow_drill.py`使用原ShadowHand候选15的合法手指闭合构型作为**固定开合模板**，仅保留手指角度，不使用该候选原手腕位姿。普通四指的屈曲关节归零得到打开状态，其余角度保留；中间状态为角度插值。根据五指末节几何中心的包围盒外扩5mm得到**暂定启发式条件描述**，不是上游人工标注过的真实内部扫掠空间；descriptor.html显示手形与两包围盒，需进一步审阅、标注和验证。

选择15是因为旧验证中其gx±仍能持续接触，不代表该模板最优。对五指手使用revolute_3f类别是现有三类接口中的多指近似；发布权重的sweep_volume_v2只读取包围盒条件，不读取完整手几何或22关节状态。输出须称为“GraspGenX基座位姿＋固定ShadowHand闭合模板”，不称为官方原生ShadowHand全关节生成。

## 坐标和尺寸

保留同一已修复电钻非凸网格及保存比例0.2554999887943268，尺寸约255.5×223.4×75mm。点云为按此比例缩放后的网格表面采样3500点，推理前减去点云均值，输出后加回。新canonical坐标Z沿原掌部Z、X从拇指指向其余四指；保存palm_from_canonical，输出再转换到DexGraspNet的掌部/WRJT坐标，不将物体比例再次乘到手腕。

## 执行命令

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

# 已准备的固定条件描述
xdg-open data/experiments/graspgenx_shadow_descriptor_20261115/descriptor.html

# 重复推理必须使用新的输出目录
shadow_ggx="data/experiments/graspgenx_shadow_drill_$(date +%Y%m%d_%H%M%S)"
/home/lee/robot_learning/open_source_baselines/GraspGenX/.venv/bin/python \
  scripts/graspgenx_shadow_drill.py infer \
  --descriptor data/experiments/graspgenx_shadow_descriptor_20261115/descriptor.json \
  --num-grasps 64 --top-k 16 --seed 42 \
  --output "$shadow_ggx"

xdg-open "$shadow_ggx/index.html"

# 使用同一原生Isaac Sim direct协议验证，非新生成步骤
shadow_ggx_check="${shadow_ggx}_isaacsim"
.venv_l20/bin/python scripts/validate_shadowhand_isaacsim.py \
  --result-file "$shadow_ggx/power-drill.npy" --top-k 16 \
  --gravity-seconds 10 --sdf-resolution 256 \
  --output "$shadow_ggx_check"
```

生成得到power-drill.npy、原始proposals.npz（canonical/palm姿态、分数、点云中心）、generation.json（输入和权重散列）、16个独立HTML及index.html。HTML是生成姿态展示，不是物理回放。energy字段仅为**负判别分数**，用于现有验证器排序；不是DexGraspNet能量，也不附加虚构E_fc/E_pen或通过标签。

本次仅授权生成，脚本不会自动启动Isaac Sim。只有完成相同物理模型/质量/摩擦/控制及严格验收的验证，才能判断新位姿是否改善稳定性。


## 本次已执行结果

`data/experiments/graspgenx_shadow_drill_20261115/`已有16候选、16独立HTML及index.html。一次采样64个，按判别分数保留16，分数范围约0.781984–0.944970。生成结果不等于稳定成功。`interface_checks.json`记录坐标与角度转换检查；CPU导入准备位于`isaacsim_inputs_check/`，不是已执行物理验证。


## 后续动力学批次已启动

用户于2026-11-15授权正式验证，输出`data/experiments/graspgenx_shadow_drill_sdf_20261115`，日志同名`.log`。16候选96方向，保持上述direct协议；与旧七候选批次物理输入逐项比较一致。启动快照尚未完成，不代表动态成功。
