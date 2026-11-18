# L20 接触点与方向修订 v4（2026-11-17）

本轮针对用户逐link核查意见，生成独立点库 `data/hand_contacts/l20_contact_v4_20261117`。旧v3及原始模型保留。当前是几何核查产物，尚未启用到抓取生成或物理验证。

## 执行的修正

|对象|处理|
|---|---|
|pinky_proximal P14/P20|法向取冻结v3蓝Zc的反向|
|pinky_distal P60/P63|删除；此前54/55/73/76继续删除，剩24点|
|middle_distal P212/P237/P227/P218|删除，剩26点|
|index_distal P297|法向取冻结v3的绿Yc正向，而不是新坐标系的绿轴|
|ring_distal|30点投影到原始STL最近表面，最大移动0.198920 mm；记录旧P、新P和Q|
|middle_middle/index_middle/thumb_metacarpals等自动点|以原始探针P到最近表面Q的方向重新估计N，闭合网格内部探针反号；棱边处仅在双侧检查通过时采用微面法向回退|
|此前人工确认方向|保留，若与内外侧检查冲突则明确标待核查|

总数420→410；累计删除稳定点号54、55、60、63、73、76、212、218、227、237。显示P号不变，新库使用独立连续runtime_id，不能把旧全局索引直接用于新数组。

## 为什么更换自动法向方法

v3的邻域面积法向簇可能选到对侧壳面。原始探针的局部距离梯度能够提供另一条方向依据，避免把所有手指一律翻转。闭合、绕向一致且体积为正的网格才执行内外判断，数值查询先转换到毫米以减少微小面容差影响。

在真实投影面Q上沿候选N正负各偏移0.05 mm，检查正向在外、负向在内。这是局部表面外向检查，不等同于指腹语义或真实受力方向；在薄壁、棱边、网格缺陷处可能不确定。T1由link +Z投影构造，T2=N×T1；每点红Xc=T1、绿Yc=T2、蓝Zc=N。

## 实际检查结果及未解决项

- 410个基均正交、右手；用户明确规则和此前人工方向保留。
- 全库191点通过双侧检查，9点冲突，210点因所属网格不闭合而没有内外认证。
- 冲突稳定号：P33、P38、P45、P50、P67、P219、P228、P231、P289。其中6点是此前人工指定方向，3点为自动估计；没有覆盖人工规则以掩盖冲突，全部标`needs_review=true`。
- ring_distal 30/30、middle_middle 20/20、index_distal 30/30、thumb_metacarpals 30/30通过；index_middle 19/20，P289需进一步核查。
- pinky_proximal网格不封闭：P14/P20已按人工要求翻转，但不能宣称由封闭体检查独立认证。
- 仅ring_distal移动，其余保留点坐标不变。原STL/URDF/原接触文件SHA保持。
- 五项方向/基向量测试通过；集成检查核对删除、映射、帧、表面投影、新库与网页坐标及66张SVG。

原始STL投影只说明回到该网格表面，不保证投影到指腹；自动方向也不自动成为有效夹持区域。下一步应优先核查9个冲突点，并区分“人工期望的宏观夹持方向”和“局部实体外法向”。之后才能固定允许区域并运行接触准入/抓取生成。

## 查看与复现

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/reports/l20_contact_frames_v4_checked_no_webgl_20261117/index.html
```

Canvas2D与静态SVG不需要WebGL；逐link表格给位置变化、方向来源和内外检查。每点原点是新版P，方向分量均在所属link坐标系中表达。

```bash
l20_v4_lib="data/hand_contacts/l20_v4_$(date +%Y%m%d_%H%M%S)"
l20_v4_view="${l20_v4_lib}_view"
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/revise_l20_contact_library.py \
  --output "$l20_v4_lib"
OPENBLAS_NUM_THREADS=1 .venv_l20/bin/python scripts/visualize_l20_contact_coordinates_2d.py \
  --contact-library "$l20_v4_lib" --output "$l20_v4_view"
.venv_l20/bin/python scripts/check_l20_contact_library.py \
  --library "$l20_v4_lib" --viewer "$l20_v4_view"
xdg-open "$l20_v4_view/index.html"
```

这些命令只重建和核查静态点库。原生成器的E_fc仍使用物体距离查询法向，本轮手部显示法向修改不会自动改变E_fc。新点坐标/删除项接入新生成任务时必须显式指定新库并重算索引与缓存，不能加载旧接触索引直接继续。
