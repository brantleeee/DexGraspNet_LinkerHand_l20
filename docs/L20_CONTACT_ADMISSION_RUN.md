# C有效接触准入：标注页与运行命令

2026-11-16。新增`admit_l20_middle_contact.py`，CPU运行，不要求CUDA、不更改A/B生成器或既有参考，不启动动力学。

## 已生成的核查页

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_review="data/experiments/l20_contact_region_review_v2_20261116"
xdg-open "$l20_review/index.html"
```

页面显示原始远节STL、211/219/228/231四点、局部坐标、投影位置、手表面最近面法向和距离。六个视角按钮按±X/Y/Z命名，不会假定哪侧是掌侧。点和网格绑定散列；法向线不是接触力。

核查步骤：旋转模型辨认计划夹持面；每点选择“掌侧/指腹允许区、指背、关节外壳、未知”，只有确认的点才勾“已核查该点”，填写依据。填写核查者，点击下载JSON。看不清保持未知，不要为了得到结果将4点全部标为允许。

下载文件名为`l20_contact_labels.json`，包含review_id、核查者、时间、逐点标签/核查状态/依据。默认模板全部未知，未代用户标注。标签是自报解剖核查，不是几何或动力学认证。

如需重新生成独立页面：

```bash
l20_review="data/experiments/l20_contact_region_review_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/admit_l20_middle_contact.py review \
  --run data/experiments/l20_contact_reselect_full_check_20261116 \
  --points 211 219 228 231 \
  --output "$l20_review"
xdg-open "$l20_review/index.html"
```

必须让下载的JSON与当前review目录配套；不同源绑定、源代码/模型/参考/代理变化会拒绝旧标注。

## 下载标注后执行C

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_review="data/experiments/l20_contact_region_review_v2_20261116"
l20_labels="$(xdg-user-dir DOWNLOAD)/l20_contact_labels.json"
l20_C_out="data/experiments/l20_contact_C_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/admit_l20_middle_contact.py apply \
  --review "$l20_review" \
  --labels "$l20_labels" \
  --threshold-mm 1 \
  --output "$l20_C_out"

cat "$l20_C_out/REPORT.md"
xdg-open "$l20_C_out/index.html"
```

若浏览器将文件另存到其他位置或自动加了后缀，修改l20_labels为实际文件路径。如果重新生成了review，也要使用该review及其下载JSON。输出目录必须新建。

## 实施的规则

1. 原静态构型admissible保留，A/B目标及原80点枚举不修改；未核查的点即便没有出现在四点页，也不会默认放行。
2. 所选点必须有palmar标签、明确reviewed=true、核查依据及核查者/时间。
3. 使用此前L20 Isaac Sim准备的同源闭合非凸代理，检查plan中的原网格/代理散列、闭合/一致绕向/正体积；由三角网格最近距离及射线包含判定外侧，而非沿用开放OBJ的原符号。
4. 内部点一律拒绝，外侧距离大于阈值也拒绝。只有数值表面容差1e-9m内作为边界。无合格组合返回no_valid_middle_contact，不回退。
5. 对合格组合沿用保存的原目标值排名；没有修改原权重、其他三槽、构型或采样。C改变可行域，要与A/B分开归因。

该代理经过原准备流程修补/实体并集，是与Sim同源的物理近似，不是实测电钻真值；本检查仅认证所选点相对于这个代理的外侧距离。它不认证完整手物穿透或整手实体自碰撞。

## 已完成的外侧预览

|构型|点|闭合代理距离mm|代理内外|
|---|---:|---:|---|
|22 行20|211|.6340|外侧|
|22 行20|219|2.3426|外侧、超过1mm|
|22 行20|228|.9327|外侧|
|22 行20|231|.8438|外侧|
|18 行43|211|.03344|外侧|
|18 行43|219|.1989|外侧|
|18 行43|228|.2718|内部、拒绝|
|18 行43|231|.3594|外侧|

实际C会对所有原合格构型中已标允许的点查询，不只检查预览两行。距离不含原sqrt(d²+1e-8)的0.1mm平滑底，因此18点211与原.1054mm显示不同。

## 输出和验收层次

- review_manifest.json：网格、点文件、代码/参考及物理代理绑定；point_info.json为局部几何；outside_preview.json为初步外侧查询；labels.template.json不含任何批准。
- C的admission_details.json：每个组合明确拒绝/接触候选准入原因，包含代理距离与内外判定。
- selected_references.json：C参考、接触点、scale1与各层准入状态；无合格时reference=null。
- REPORT.md/index.html：状态、构型、点号、外侧距离和合格C静态页。

contact_candidate_admitted只表示该点的区域自报核查、外侧距离及原可行域通过；static_geometry_admitted、execution_admitted和dynamics_admitted仍为false。真实绝对几何、开放手部包含、闭合路径和实际接触执行未完成，不自动开展600g六方向验证。

## 测试范围

3项准入测试通过：所有条件同时满足才放行，闭合盒内/外同距离不能混淆，资产修改后旧绑定拒绝。与2项接触重选测试合计5项通过。真实资产核查页及8个预览查询已完成，HTML控件/载荷绑定检查通过；浏览器人工交互尚待用户使用。

全unknown标注的负向集成检查位于l20_contact_C_negative_v1_20261116：两个候选均no_valid_middle_contact、无参考回退、无动力学。这是自动拒绝测试，不是人工标注或C成功实验。尚未虚构任何真实点的palmar核查。

## v2投影修正

2026-11-16修正米制微面最近点容差，增加选面/切向局部证据。请使用v2页面重新核查、下载标注；旧v1标签不兼容。详情见[L20_NORMAL_REVIEW_FIX.md](L20_NORMAL_REVIEW_FIX.md)。历史v1负向测试仅为当时检查，不代表v2正向认证。
