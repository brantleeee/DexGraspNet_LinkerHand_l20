# L20 人工方向修订 v6（2026-11-17）

最终库：`data/hand_contacts/l20_contact_v6_confirmed_20261117`。页面：`data/reports/l20_contact_frames_v6_confirmed_no_webgl_20261117/index.html`。旧v5与本轮早期未包含中指回复的试算库保留，最终请使用带confirmed的目录。

## 明确修改

所有轴选择都引用**同一份冻结v5**，不会在已修改的坐标系上再次旋转。

|对象|处理|
|---|---|
|ring_distal P133|新版N=−v5.Yc，即旧绿轴反向|
|ring_distal P139、P150|删除|
|middle_distal P223|删除；用户后续红轴点号也包含223，但没有明确撤销删点，故仍删除|
|middle_distal P211、P219、P235、P231、P230|新版N=v5.Xc，即旧红轴正向；用户未另注明反向，按其“红线指向”执行|
|thumb_metacarpals|全部30个保留点N=−v5.Zc，包含P337|

403点、累计17点删除。所有其余点位保持v5原值；未指定的20个中指点不自动改变方向，也不因“未指定”就判为错误。红/绿切向随新N重新构造，始终保持Xc×Yc=Zc；**新版蓝轴才是指定后的法向**，旧红/绿轴的颜色不随解释混用。

## 方向与几何证据必须分开

人工选择表达用户期望的方向，不能继承旧N的外向认证。所有36个本轮改向点保存旧框架/旧质量记录，将当前`outward_probe_pass`置为未知。橙色三角面可能仅是历史参考，质量字段给`support_role=historical_reference_only`；P133的新N与旧支撑面可能不垂直，拇指翻转也可能与旧网格外向相反，不能继续声称“蓝轴垂直橙面且已外向认证”。原始geometric_N独立保留，没有改STL来迎合箭头。

因此本轮结论是**已精确应用人工方向规则**，不是全部接触面/夹持区域或物理稳定性已经认证。默认生成器和旧实验没有替换。

## 查看

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/reports/l20_contact_frames_v6_confirmed_no_webgl_20261117/index.html
```

无WebGL；单点三轴、历史参考面、P–Q线、坐标与来源表格保留。额外的`middle_distal_axis_review.html`只用于后续核查：图恢复冻结v5轴、表格是当前记录，黑箭头预览所选N；可记录点号/轴并下载含源散列的JSON，不会在浏览器自动修改源文件。

## 复现

```bash
l20_review_lib="data/hand_contacts/l20_review_$(date +%Y%m%d_%H%M%S)"
l20_review_view="${l20_review_lib}_view"
.venv_l20/bin/python scripts/apply_l20_contact_review.py \
  --middle-red-ids 211 219 235 231 230 --output "$l20_review_lib"
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/visualize_l20_contact_coordinates_2d.py \
  --contact-library "$l20_review_lib" --output "$l20_review_view"
.venv_l20/bin/python scripts/build_l20_axis_review.py --viewer "$l20_review_view"
```

若使用核查页下载记录，将`--middle-red-ids ...`替换成`--review-json /实际下载路径/l20_middle_v5_axis_review.json`，仍输出到新目录。应用器只接受散列匹配的冻结v5和有效middle_distal保留点号。

5项单测通过，覆盖右手重建和旧外向标签失效；集成检查403点、36条精确改向、删点、源SHA、连续runtime、其他框架/点位不变及网页一致。Canvas2D模拟DOM完成选择、黑箭头预览、记录与JSON下载检查。未进行动力学或新抓取生成。
