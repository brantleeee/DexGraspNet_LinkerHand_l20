# 每个接触点的坐标系：L20，无WebGL（2026-11-17）

上一版仅显示各link原点轴及每点XYZ，缺少每个点自身的三轴；本轮补齐。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/reports/l20_every_contact_frame_no_webgl_20261117/index.html
```

22个link、420个点，每个点P都有自己的坐标系，原点严格位于原候选P，而不是表面投影Q或link原点。

- 默认接触基：Xc=T1（红）、Yc=T2（绿）、Zc=N（蓝），右手，方向均用该link局部坐标表达。原link原点/轴仍单独保留。
- 可切换为“每点平行link XYZ”：各点各自的原点P，三轴平行父link，用于仅核查坐标轴位置。
- “每点坐标系”默认开启，轴长默认2mm可调；下拉单独检查某点，标签/网格可关闭，拖动旋转及XY/XZ/YZ保留。
- 66张静态SVG点集投影也包含每点三轴；投影某轴可能缩成一点，不能据此判三轴不正交。
- all_contact_frames.json及每link_frames.json导出每点原点、Xc/Yc/Zc、父link、原微面N及法向来源；原坐标CSV不变。

法向默认取原STL最近微面，尚未认证实际接触方向；中指211/219/228/231沿用用户确认覆盖。切向不代表滑移方向，未将任何点或区域自动批准。

复现：

```bash
.venv_l20/bin/python scripts/visualize_l20_contact_coordinates_2d.py \
  --output "data/reports/l20_every_contact_frame_no_webgl_$(date +%Y%m%d_%H%M%S)"
```

验证420个原点与原点坐标逐一一致，420基矩阵正交且右手，66SVG解析通过；V8模拟无WebGL环境实际执行Canvas2D，逐点三轴标签/箭头绘制检查通过。没有修改原网格、点文件或抓取结果。
