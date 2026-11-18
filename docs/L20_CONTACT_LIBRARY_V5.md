# L20 接触坐标系 v5：明确支撑面（2026-11-17）

本轮针对thumb_metacarpals P337、middle_distal/ring_distal框架及食指中节删点。输出独立库`data/hand_contacts/l20_contact_v5_20261117`，保留v4和旧实验。

## 修改与证据

- index_middle删除稳定号P271、P273、P280、P284，全库410→406点，runtime_id重新连续编号。
- middle_distal 26点、ring_distal 30点重新绑定最近支撑三角面。法向直接取该面的外法向，不再将棱边的点到面方向当作该三角面的垂直法向。
- 距离相同（容差0.00001mm）的面，确定性选择面积最大的三角面；保存并标记具有明显不同法向的等距面，不把选面规则当作夹持面认证。
- 本轮重建两个远节全部框架，包括先前人工方向；旧方向与来源保存在新记录previous_Zc/previous_normal_source以及旧v4库。其他link的人工方向均保留。
- 所有保留点的位置与v4完全相同。ring_distal沿用v4的投影位置，本轮不再次移动点。
- thumb_metacarpals P337新增支撑面证据。该点距网格约0.102542mm，最近投影位于三角面内部，旧蓝轴与该面法向一致，新旧N夹角约0°；不能据当前证据宣称“已翻转修好”。如果用户指的是夹持面选错，需要进一步明确目标表面。

## 为什么v4内外检查不足

最近点Q落在棱边时，P−Q可能落在相邻面的法向锥中，而非垂直某个具体三角面；v4沿该方向检测正外负内，仍可能通过。本次实际检查中，中指/无名指远节多点与最近面法向差15°以上。v5把N绑定到明确面，保留选面不唯一的证据。

57个绑定面均通过三角面中心正负0.01mm的内外检查，但32点的最近表面有不同法向的等距面。面中心检查仅用于方向一致性，不能替代P处的实际接触或区域认证。

## 查看

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/reports/l20_contact_frames_v5_no_webgl_20261117/index.html
```

选择middle_distal/ring_distal的单个P点，或thumb_metacarpals的P337：橙色三角形为实际支撑面，紫线连接P和投影Q，蓝Zc=N垂直橙面；红Xc/绿Yc为切向。每点三轴仍以P为原点、在所属link局部系表达。全部点模式不叠加支撑面以免遮挡。Canvas2D不依赖WebGL，另有66张静态SVG。

下一次反馈请区分：蓝N未垂直橙面、橙面选错、P位置不合适、仅切向方向希望调整。棱边不存在唯一表面法向，因此需要先确定实际计划接触哪一侧。

## 复现与验证

```bash
l20_v5_lib="data/hand_contacts/l20_v5_$(date +%Y%m%d_%H%M%S)"
l20_v5_view="${l20_v5_lib}_view"
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/rebuild_l20_attached_frames.py \
  --output "$l20_v5_lib"
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/visualize_l20_contact_coordinates_2d.py \
  --contact-library "$l20_v5_lib" --output "$l20_v5_view"
.venv_l20/bin/python scripts/check_l20_attached_frames.py \
  --library "$l20_v5_lib" --viewer "$l20_v5_view"
```

7项单测通过，含尖锐棱边与附近对侧面负向例。集成核对406索引、原始源SHA、其他框架不变、点位置不变、57面垂直性/右手正交/局部外向及网页数据一致。V8模拟DOM执行Canvas支撑面、Q及Zc/N标签绘制通过，未声称在用户浏览器实测。

本轮没有优化抓取、训练或动力学实验，默认生成器未启用新库，E_fc的物体查询法向未修改。几何框架正确不等于夹持区域正确或抓取稳定。
