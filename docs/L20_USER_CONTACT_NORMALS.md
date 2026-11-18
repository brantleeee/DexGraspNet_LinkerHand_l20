# 用户确认的中指接触方向覆盖（2026-11-16）

依据用户对`l20_contact_middle_distal_frame_20261116`主图的确认：219/228的接触法向取该图−T1，211/231取该图T2。规则只相对于这份冻结坐标基，不相对于每次重新计算后的新T1/T2，避免重复旋转。

覆盖配置：`grasp_generation/configs/l20_middle_user_contact_normals_20261116.json`。绑定原contact_frames.json散列、手部URDF、STL及接触点文件。保存原N/T1/T2、规则和显式新N，载入时验证向量确实满足原规则及资产未变。

|点|新N依据|新N（middle_distal局部系）|
|---|---|---|
|211|原T2|(.29450,−.26220,.91898)|
|219|原−T1|(.29413,−.26287,.91890)|
|228|原−T1|(.34785,.00092,.93755)|
|231|原T2|(.45688,−.25293,.85281)|

新T1由原微面N投影到新切平面并归一化，新T2=N×T1，维持T1×T2=N。原STL/微面法向不修改，contact_frames.json分别存geometric_N及用户确认N，normal_source明确标识。主图红N采用覆盖方向，旧微面N_facet用灰虚线可在图例打开；下方三角面局部证据仍保留真实原微面N，不要将两者混淆。

打开已生成页面：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/experiments/l20_contact_user_normals_20261116/index.html
```

重新生成：

```bash
.venv_l20/bin/python scripts/admit_l20_middle_contact.py review \
  --normal-overrides grasp_generation/configs/l20_middle_user_contact_normals_20261116.json \
  --output "data/experiments/l20_contact_user_normals_$(date +%Y%m%d_%H%M%S)"
```

如继续C区域标注，使用对应新页面下载JSON和该目录运行apply；覆盖法向本身不会自动填写palmar或放行几何/路径。

3项基向量测试及四点与原显示方向的数值比对通过，正交/右手性通过。这是用户指定的手部接触框架覆盖，不是STL单面法向修复或仿真成功。

生成器E_fc使用物体距离查询返回的法向，和这里的手部有效接触方向不是同一个变量。本次未替换E_fc或改物体SDF；若后续引入手物接触方向一致性项，须在新实验中明确两个法向的坐标变换及适用接触区域，不能把本覆盖宣称为生成器法向已经修复。
