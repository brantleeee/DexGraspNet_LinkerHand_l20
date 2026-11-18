# L20各link接触点XYZ：无需WebGL（2026-11-17）

对象是LinkerHand L20，非ShadowHand。脚本`visualize_l20_contact_coordinates_2d.py`生成22个link页面、420个接触点。原STL、点文件及抓取/法向都不修改。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/reports/l20_link_coordinates_no_webgl_20261117/index.html
```

## 页面操作

- 主图使用Canvas 2D软件正交投影，不加载Plotly，不申请WebGL上下文，也不fetch数据。拖动旋转、滚轮缩放、3D/XY/XZ/YZ按钮切换。
- 点旁直接显示global点号及(x,y,z)mm；下拉可只看一个点，XYZ标签和网格可分别关闭。坐标是该link局部系，未转腕/世界坐标。
- 点叠加显示、不做网格深度遮挡，背面点也可见；用于核对坐标，不能凭遮挡判断真实接触。
- 下方三张SVG是点集XY/XZ/YZ静态投影，标签含完整XYZ，故无JavaScript时也能查看；静态投影不显示网格，不能当碰撞证据。
- 每link有坐标表和CSV/JSON，整个目录另有all_contact_coordinates.csv/json。空接触link明确标记，不遗漏。
- 保留整个目录，HTML使用本地viewer.js及SVG。没有外部CDN、WebGL或浏览器GPU要求。

重新生成：

```bash
.venv_l20/bin/python scripts/visualize_l20_contact_coordinates_2d.py \
  --output "data/reports/l20_link_coordinates_no_webgl_$(date +%Y%m%d_%H%M%S)"
```

CPU运行，新目录；不是生成抓取或动力学命令。

## 检查证据

22页、420点、66份SVG存在且XML可解析；420点与原L20 JSON同link/local索引匹配，float32载入容差1e-8m，米/毫米一致，global ID唯一连续。Canvas JS经V8语法及无WebGL模拟DOM执行验证，可完成三角网格、坐标轴和XYZ标签绘制；未声称已在用户浏览器实际交互验收。无Canvas时坐标表及独立SVG仍可用。
