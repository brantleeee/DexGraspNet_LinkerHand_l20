# 官方ShadowHand各link接触点XYZ检查（2026-11-17）

新脚本`visualize_shadowhand_contact_coordinates.py`直接加载本地官方HandModel与接触点，不改模型、点或法向。输出每个PK link计算框架的独立页面；包括无几何虚拟框架和无接触点link，明确区分，不将它们当额外硬件连杆。

已完成：45个link/计算框架页面，其中14个有接触点，共140点。图中每点直接显示全局ID和(x,y,z)，默认标签毫米；坐标轴及*_m数据始终为米，悬停/表格同时给米和毫米。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/reports/shadowhand_link_coordinates_20261117/index.html
```

目录分为有点、无点两组。每页可用六视角旋转、下拉单独查看某点，坐标表和CSV/JSON同时提供全局点号、link内local点号、局部XYZ。坐标属于标题中的官方link（例如robot0:mfdistal_child），没有转换成腕/世界系，不同link数值不可不经FK直接比较空间距离。

整体导出为all_contact_coordinates.csv/json，每link单独一份。plotly.min.js共享一次，拷贝页面时保留整个目录。显示胶囊按官方解析段摆放，延续此前运行库显示网格偏移的修正；不修改真实点数据。

重新生成：

```bash
.venv_l20/bin/python scripts/visualize_shadowhand_contact_coordinates.py \
  --label-unit mm \
  --output "data/reports/shadowhand_link_coordinates_$(date +%Y%m%d_%H%M%S)"
```

--label-unit m改成米制文字标签。CPU运行，新目录，不进行抓取生成或仿真。

验证：45页及本地JS存在，140全局索引连续唯一，逐点与官方JSON同link/local索引匹配（float32载入容差1e-8m），米/毫米换算、图上XYZ标签及数据文件检查通过。没有新增物理结论。
