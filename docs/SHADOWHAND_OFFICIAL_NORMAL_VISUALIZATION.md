# 官方DexGraspNet ShadowHand接触点法向核查

2026-11-17。用户要求对官方ShadowHand可视化，判断L20显示中的方向问题是否也存在。

## 来源与定义

直接加载本地`external/DexGraspNet_official/grasp_generation/utils/hand_model.py`，断言实际模块路径；读取官方shadow_hand_wrist_free.xml、contact_points.json，没有使用L20模型或法向覆盖，也不改官方源。官方本地仓库HEAD为bd1b13d7248729af117e1d46aaa6266b147a3c7b；页面manifest记录关键源文件SHA及实际库版本。

官方contact_points.json仅存坐标，不存手部法向；energy.py的E_fc从object_model.cal_distance取得物体距离查询法向。本图不能认证E_fc或物体法向正确。

官方接触所在14个连杆均为简化胶囊碰撞模型。根据官方cal_distance的线段z∈[0,2h]与半径r，定义轴上最近点c、外法向N=(p−c)/||p−c||、表面点Q=c+rN。原函数正内SDF为r−||p−c||，其梯度是−N，而图中红N为向外方向；不要把符号差当错误。T1取局部Z投影到切平面（必要时回退），T2=N×T1，二者只是基向量。

## 已生成页面

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/reports/shadowhand_official_normals_v1_20261117/index.html
```

共有14个接触连杆、140点。五指远节各10点：

|页面|部位|
|---|---|
|ffdistal_child.html|食指远节|
|mfdistal_child.html|中指远节|
|rfdistal_child.html|无名指远节|
|lfdistal_child.html|小指远节|
|thdistal_child.html|拇指远节|

全部点默认只显示红N，下拉可显示所有N/T1/T2或单点。每页当前child局部坐标轴、全局点号/连杆内local号及完整数值JSON均提供。点号属于官方140点集合，与L20的420点集合不同。

复现命令（CPU、不需要CUDA）：

```bash
.venv_l20/bin/python scripts/visualize_official_shadowhand_normals.py \
  --output "data/reports/shadowhand_official_normals_$(date +%Y%m%d_%H%M%S)"
```

加入--distal-only只导出五指远节50点。输出目录必须新建。

## 发现的显示/解析几何差异

在当前Trimesh4.5.1中执行原HandModel的胶囊构造后，远节显示三角网格的Z范围与其cal_distance解析胶囊段不同：普通四指bounds最大差约12mm，拇指约13mm。官方代码的Capsule生成、−h平移与visual.offset组合，在当前运行库下得到中心化网格，而解析SDF仍使用0..2h段。

因此主图灰色网格专门放在解析胶囊位置，与红N定义相同；橙色原HandModel运行库网格保留为可开关对照，默认隐藏。若打开橙色后发现红N不垂直其表面，不能直接归为官方预存法向错误。该差异也不自动证明当年官方实验使用的库存在相同问题，本轮未修改或运行旧官方环境。

五指远节官方点距解析胶囊表面的最大偏差均<.01mm。该结果说明这些点贴近当前官方解析简化几何；不代表真实硬件接触法向或动力学已认证。L20使用原STL微面，官方使用简化胶囊，两者表面复杂度不同，不应只比箭头外观。

## 验证

2项解析胶囊测试通过：侧面/端帽外法向与表面位置、正内SDF梯度=−N的有限差分。全部14页/140基向量的单位、正交、右手性、五个远节合计50点、实际官方模块路径验证通过。未启动生成、物体测试或动力学，也没有给官方E_fc作正确性结论。
