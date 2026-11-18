# L20新版接触点库与面片坐标系（2026-11-17）

用户指出小指中/远节方向及不合适点。新版库独立保存，不覆盖旧contact_points.json或旧实验：`data/hand_contacts/l20_surface_patch_v3_20261117`。

## 已执行的明确修改

- 删除稳定点号P54、P55、P73、P76，420→416点，小指远节30→26点。
- P33法向取旧页面红Xc正向。
- P50、P65、P74、P79法向取旧红Xc反向。
- P67法向取旧绿Yc正向。
- 中指211/219/228/231保留此前确认方向。

旧页面固定为`l20_every_contact_frame_no_webgl_20261117/all_contact_frames.json`，配置保存其SHA。所有方向相对该冻结基，不相对重建后的新轴，避免重复旋转。

## 其余坐标系重建方法

原方案只取单个最近微三角面，遇到微小侧壁、内壳或棱边会偏离宏观接触面。新方法取最近表面点2mm邻域，按面片面积和距离加权，对35度法向簇选择主面片方向；同link中8mm内存在用户确认点时，以其方向为60度范围内的簇选择引导。这些是待验证的估计参数，不是硬件标定。

切向不再跟随任意微三角边：将link +Z投影到新切平面作为T1（退化时回退），T2=N×T1；显示仍Xc=T1红、Yc=T2绿、Zc=N蓝。所有坐标系原点保持原候选P，没有将点移动到投影面。

自动估计仍标`surface_patch_estimate_unreviewed`，记录主簇面积支持比例、是否冲突、引导点及距离；不会自动成为有效夹持区域或准入。P37、P49借P33引导后法向候选约(.856,−.501,.130)、(.844,−.518,.140)，但支持比例约.189/.225，仍属低置信度，必须重点看图，而非宣称已完全修好。其他无人工依据的406点也保持待核查。

## 查看

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/reports/l20_contact_frames_v3_no_webgl_20261117/index.html
```

仍使用Canvas2D，不需要WebGL；可直接开pinky_middle.html及pinky_distal.html，蓝Zc是新版法向。每点方向来源/支持比例/待核查状态在表格和帧JSON中。

页面P编号保持旧稳定点号便于反馈；新生成库的连续索引记录在runtime_id中。id_mapping.json给出stable_id→runtime_id及删点项；不得把显示P79直接当新版生成索引79。新版contact_points.json已物理删除4点，并保持其他原始坐标。

## 复现

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_lib="data/hand_contacts/l20_patch_$(date +%Y%m%d_%H%M%S)"
l20_view="${l20_lib}_view"
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/rebuild_l20_contact_frames.py \
  --patch-radius-mm 2 --anchor-radius-mm 8 --output "$l20_lib"
.venv_l20/bin/python scripts/visualize_l20_contact_coordinates_2d.py \
  --contact-library "$l20_lib" --output "$l20_view"
xdg-open "$l20_view/index.html"
```

## 文件与验证边界

- contact_points.json：新版生成可用的416坐标，仅删除不移动；旧生成器默认仍原文件，须在未来新任务显式接入新库，不能悄悄覆盖旧实验。
- contact_frames.json：新版法向/切向、稳定与连续索引、人工规则/自动质量。
- id_mapping.json：包含删除及重编号的完整420源点映射。
- manifest.json：源模型/参考/算法及生成文件散列。

5项面片/基向量测试通过；416点位置、6项新增人工方向、4项保留方向、删除项、连续runtime索引、416右手正交基、页面/JSON数目核对通过。原网格/旧点文件未改，未修改生成器使用的物体法向或E_fc，未启动抓取优化或动力学。显示重建不等于真实接触或承载已合格。
