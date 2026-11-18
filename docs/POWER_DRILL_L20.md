# 电钻OBJ：LinkerHand L20抓取生成

更新：2026-11-13。输入：`assets/power-drill-obj.zip`；物体代码/别名均为`power-drill`。

前16候选的MuJoCo原生SDF验证见[动力学验证说明与命令](POWER_DRILL_MUJOCO_SDF.md)。

## 已完成准备

`scripts/prepare_power_drill.py`直接读取压缩包指定OBJ，保留原OBJ/材质/纹理及CC0许可证到`data/meshdata/power-drill/source/`。焊接重合接缝、删除重复/退化三角面；原模型有一段环形零件缺少内壁，通过两圈可一一对应的边界补80个内壁三角面，保留中心孔而非封闭端口。对25个零件做manifold实体并集，去除内部重叠表面，输出封闭、绕序一致的非凸实体（5058顶点、10112面）。未做全物体凸包、凸分解或降面近似；并集/补壁仍属于几何处理，不能称逐面原样模型。

输出`coacd/decomposed.obj`只是兼容旧加载器的文件名，并非CoACD处理结果。原数据与处理步骤、输入/输出SHA256记录在`import.json`，后续生成和验证共用该几何和坐标。已在物体目录注册，后续Sim入口支持`--object power-drill --result-file 新结果`，不会误用杯子旧位姿。

## 尺寸与坐标

源包没有明确长度单位，原始包围盒为0.2555×0.2234×0.075。准备后将包围盒中心平移到原点，保持源轴方向，最长边归一化为1。

因此生成时的`--scales`数值就是最长边的米数：

|scale|实际包围盒尺寸（mm）|
|---|---|
|0.20|200×174.873×58.708|
|0.25|250×218.591×73.386|
|0.2555|255.5×223.4×75|

不要直接沿用杯子的0.06：这里会得到最长边60mm的微型电钻。手不缩放；每个scale独立初始化与优化，保存qpos平移已匹配其比例，后续不再乘一次scale。

## 正式生成命令

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

# 已准备好时复核散列并复用，不覆盖原始资产。
.venv_l20/bin/python scripts/prepare_power_drill.py

l20_drill_name="l20_power_drill_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/generate_small_grasps.py \
  --object-code power-drill \
  --scales 0.20 \
  --batch-size-per-scale 16 \
  --n-iter 6000 \
  --hull-inflation scale-aware \
  --name "$l20_drill_name"

xdg-open "data/experiments/$l20_drill_name/results/scales/scale_0p2/power-drill_vis_best.html"
```

此命令针对L20右手生成16个候选，6000是静态优化迭代数，不是RRL训练轮数。当前在整把电钻表面生成，不限定只抓握柄。

若需两种尺寸，改为`--scales 0.20 0.25`，每尺寸16个，共32个候选；对应结果分别在`results/scales/scale_0p2/`和`scale_0p25/`。可以用相同物体代码、不同实验名重复实验，不覆盖旧结果。

## 输出与验收边界

- `results/power-drill.npy`：本批全部候选。
- `results/power-drill_scales.json`：比例、源网格及分组信息。
- `results/scales/scale_0p2/power-drill.npy`：该比例候选与匹配qpos。
- `results/scales/scale_0p2/power-drill_vis_best.html`：最佳几何候选静态可视化，未过阈值时也可能显示最低能量候选。
- `scale_validation.json`：URDF可行性与静态几何筛选计数。

已实际执行`l20_power_drill_smoke_20261113`：2候选、2优化迭代，比例/有限值及URDF可行性2/2通过，静态几何阈值0/2，HTML输出成功。该短检查只证明链路可用，不能当作有效抓取结果；正式6000迭代及动力学稳定性尚未验证。另已通过内壁修复保持孔体积的CPU测试，以及Sim输入准备检查，未启动Sim验证。

后续应先查看静态接触/穿透指标，再用当前六方向动力学协议验证；不能把生成结束、HTML看似贴合或最低能量排名直接等同于稳定抓取。

## 恢复源OBJ尺寸，并保存能量排名前16的独立HTML

如果将源OBJ坐标单位按米解释，“不额外缩放”是恢复255.5×223.4×75mm的源尺寸。由于预处理已最长边归一化，需指定`--scales 0.2555`抵消该归一化；`--scales 1`会变成1米长的电钻。几何仍为已修复/并集模型，并保留居中变换，不是未处理的原坐标直接导入。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

.venv_l20/bin/python scripts/prepare_power_drill.py

l20_drill_name="l20_power_drill_original_size_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/generate_small_grasps.py \
  --object-code power-drill \
  --scales 0.2555 \
  --batch-size-per-scale 32 \
  --n-iter 6000 \
  --hull-inflation scale-aware \
  --export-top-k 16 \
  --name "$l20_drill_name"

xdg-open "data/experiments/$l20_drill_name/results/scales/scale_0p2555/top_16/index.html"
```

生成32候选，按保存的总能量升序选16个，能量并列按源数组顺序；输出`rank_01_candidate_XXX.html`至`rank_16_candidate_XXX.html`，各HTML自带Plotly，可独立打开。`ranking.json`保留原下标、scale、能量及几何阈值是否合格。没有只保留PASS：几何FAIL也可能进入能量排名，标题及索引页会显示，不把能量排名当稳定抓取排名。

已有结果可单独导出，无需重跑生成：

```bash
.venv_l20/bin/python scripts/export_top_grasps.py \
  --object-code power-drill \
  --result-file "data/experiments/$l20_drill_name/results/scales/scale_0p2555/power-drill.npy" \
  --top-k 16 \
  --output "data/experiments/$l20_drill_name/ranked_export_new"
```

新目录保护避免覆盖旧HTML。4项CPU排序测试通过，且已在2候选短检查数据上实际导出两个独立HTML；尚未执行这条32候选/6000迭代正式命令。
