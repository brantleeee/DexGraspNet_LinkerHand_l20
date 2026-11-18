# 独立进程复现、逐指诊断与局部参考细化

2026-11-06。保留原杯子0参考和v4/v5检查点；不根据同进程历史下的旧标签直接训练。

## 已完成实测

|扰动来源（20样本表）|两个新进程的零残差结果|完整13秒过程复现|结论|
|---|---|---|---|
|组18|两次均6/6|位置/关节/目标/力差均0|没有该样本的参考失败可细化|
|组3|两次均4/6|位置/关节/目标/力差均0|gy+、gz+是当前可复现失败|

四元数角度比较残差约4e-6°来自归一化/浮点计算，远低于过程诊断阈值。
这些事实支持独立进程排除本次同进程历史效应，但不证明warm-start是唯一根因或所有样本都可重复。
每个进程只构建一次场景、只跑一回合；每回合仍六环境，保持原克隆布局及方向映射。
过程诊断容差独立于5mm/15°/1mm等物理抓取标准，不放宽成功条件。

组3首次失败：gy+全局6.90417秒（重力后3.90417秒），gz+全局3.05秒（重力后50ms）。
开重力时逐指法向力标量和（N，SDK数据）：

|方向|拇指|食指|中指|无名指|小指|掌部|
|---|---:|---:|---:|---:|---:|---:|
|gy+|0.07285|0|0|0.08422|0.02453|0.16532|
|gz+|0.02132|0|0|0.02395|0.01298|0|

食指存在近接触记录但法向力为零，中指无有效法向受力；不是所有“contact count>0”都代表夹持。
gz+的无名指受力在重力后约4.17ms进入持续间断，之后50ms滑移超限。
这些记录提示接触分布不足，尚不能单凭总力断定力闭合或唯一失败机制。

食指PIP局部试验 `-0.015/+0.015 rad`（分别两个新进程）都保持4/6，原四个成功方向保留，没有合格改善候选。
+0.015时gy+提前到重力后54.2ms失败，gz+推迟至279.2ms，存在方向取舍，不能将“延迟掉落”当作成功。
未进入RRL训练，也未导出虚构的改善reference。

## 接触可视化与数据

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
xdg-open data/experiments/mug0_isolated_group03_20261106/zero_a/contacts/index.html
cat data/experiments/mug0_isolated_group03_20261106/REPORT.md
cat data/experiments/mug0_isolated_group03_20261106/zero_a/contact_events.json
```

图中有逐指法向力、物体运动、接触点及法向3D视图。失败时无点则显示此前最后有点的时刻，标题标明实际快照时间。
`trajectory.npz`保留240Hz逐连杆/逐指力、接触数、关节和物体状态。
`contact_points.npz`是无截断的ragged记录：物理步、环境、连杆索引、点、法向、法向力、间距；摩擦锚点及向量单独记录。
坐标均还原到原物体坐标方向（环境旋转前）；接触法向保留SDK符号，未擅自标为从物体指向手。
摩擦锚点不与法向点一一对应。接触脱离事件要求标量法向力>0.001N且有点，间断至少12帧/50ms才记录，诊断不修改成功标准。

## 复现检查入口

```bash
l20_audit="data/experiments/mug0_isolated_g03_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/check_isaacsim_isolated.py \
  --samples data/experiments/mug0_v5_half_confirm20_20261106_135255/perturbations.json \
  --group 3 --output "$l20_audit"
```

两个子进程串行执行，自动检查状态、过程与标签一致性，生成图表。
可加 `--checkpoint ... --preload-gain 0.5`：只有零–零复现通过后才在第三个新进程评估原策略，并审计2.5秒前缀。
所有原生Sim进程共用锁，已有任务会排队，不强制结束已有进程。

## 下一步：一个联合关节候选，保持手腕不变

食指单独调整未修复，下一候选检验同时建立食指/中指接触；这是待验证建议，不保证改善。

```bash
l20_joint="data/experiments/mug0_local_index_middle_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/refine_isaacsim_reference_local.py \
  --audit-results data/experiments/mug0_isolated_group03_20261106 \
  --joint-deltas index_pip=0.02 middle_pip=0.02 \
  --output "$l20_joint"

cat "$l20_joint/REPORT.md"
cat "$l20_joint/results.json"
.venv_l20/bin/python scripts/plot_isolated_contacts.py --run "$l20_joint/coupled_joints/a"
xdg-open "$l20_joint/coupled_joints/a/contacts/index.html"
```

只测试一个联合候选，两个新进程各13秒六方向；相同对象扰动、质量、摩擦、PD、SDF、mimic及严格验收不变。
自动拒绝不复现或没有失败的基线。候选须满足URDF/mimic硬约束、不损害原通过方向且增加通过数，才能进入best_reference。
每关节偏置≤0.03rad，不直接优化被动DIP；从动关节按mimic同步，scale与手腕不二次缩放。
默认不调整手腕。可用 `--joints ... --delta-rad ... --max-joint-proposals ...` 做有预算的坐标搜索。
只有显式 `--allow-wrist` 才在关节候选未增加通过数时试局部±1mm/±1°位姿，允许上界2mm/2°；有限关节搜索无改善不证明关节空间不可修复。
手腕变换同步所有手部刚体和世界固定锚点，物体不动；已做USD几何/质量/惯量/锚点不变量短检查，未声称腕部动力学候选已验证。

## 保存改善参考，再考虑RRL

如果summary.json的selected_run为null，停止，不能执行导出或假装已有改善。
若找到候选，best_reference/reference.json与validation.json会记录新目标、父参考、实际验证目录与仅单样本的验证范围。
先用其他平移在独立进程验证不损害。示例：组18原参考已复现6/6，候选也应保留六方向通过；这里只在确实生成best_reference后执行：

```bash
l20_cross="data/experiments/mug0_local_cross_g18_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/check_isaacsim_isolated.py \
  --samples data/experiments/mug0_v5_half_confirm20_20261106_135255/perturbations.json \
  --group 18 --variant-reference "$l20_joint/best_reference/reference.json" \
  --output "$l20_cross"
```

此跨样本入口仍是两个全新零残差进程，不允许载入旧参考策略。通过后再将新来源单独导出：

```bash
.venv_l20/bin/python scripts/export_refined_grasp.py \
  --bundle "$l20_joint/best_reference" \
  --output "$l20_joint/derived_grasps"
cat "$l20_joint/derived_grasps/EXPORT.json"
```

新NPY保留原数组索引/scale，仅替换已选候选qpos；旧静态能量移至parent_static_metrics，不能冒用旧E_pen做官方Gym筛选。
原NPY、场景和检查点均不覆盖。后续可把EXPORT.json中的result_file送入现有batch_l20_isaacsim.py重新构建/验证新来源，并通过其新reference-results训练独立策略。
新模型构建不替代本独立进程的可信比较；新来源重新进入RL之前，须重复零残差及共同前缀检查。
若改善参考已在所测条件全通过，不能再宣称失败修复训练有必要；RRL任务应另有明确扰动缺口，不直接加载旧候选策略。

## 工程验证范围

CPU测试45项通过；原v4/v5训练代码散列仍匹配。
真实运行：组18两进程、组3两进程、食指PIP两个候选各两进程，共8个单回合原生Sim进程，不训练。
另做一次USD几何短检查（不模拟回合）。联合食指/中指候选与跨样本确认尚未运行。

## 运行时无输出的排查

后续入口已直接转发简洁进度，完整原生日志仍保存到候选a.log/b.log；worker状态区分初始化/仿真/保存/完成。
旧已启动进程可能仍安静，请在另一终端读取其输出目录下 `coupled_joints/*.log`；报告只有两轮和汇总结束才生成。
共享运行锁意味着重复启动会排队，不要因终端安静而用新时间戳反复启动。进程/日志只读快照不等于持续运行监控。

## 联合PIP结果与下一个限定候选

联合食指/中指PIP各+0.02rad已测，两个新进程仍4/6；gy+失败从3.904秒提前至0.254秒，gz+仅50→79ms，目标两指法向力仍0，不能保留为改善。
下一候选从原参考出发，只改两指MCP屈曲，不累加失败PIP修改：

```bash
l20_mcp="data/experiments/mug0_local_mcp_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/refine_isaacsim_reference_local.py \
  --audit-results data/experiments/mug0_isolated_group03_20261106 \
  --joint-deltas index_mcp_pitch=0.015 middle_mcp_pitch=0.015 \
  --output "$l20_mcp"
cat "$l20_mcp/REPORT.md"
cat "$l20_mcp/results.json"
.venv_l20/bin/python scripts/plot_isolated_contacts.py --run "$l20_mcp/coupled_joints/a"
xdg-open "$l20_mcp/coupled_joints/a/contacts/index.html"
```

CPU计划通过，实际未运行。若仍无有效受力，先量化几何间隙与接近方向，再决定手腕细化，暂不继续小步搜索或RRL。

## MCP结果已修复组3，下一步跨组18

`mug0_local_mcp_20261106_163701` 两个新进程均6/6，已保存best_reference。
中指仍全程无接触，但食指和掌部受力显著增加；当前任务不要求每指接触。联合改变两个关节，未分离其各自贡献。
冻结候选，先验证原参考已成功的组18是否受损：

```bash
l20_mcp_cross="data/experiments/mug0_mcp_cross18_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/check_isaacsim_isolated.py \
  --samples data/experiments/mug0_v5_half_confirm20_20261106_135255/perturbations.json \
  --group 18 \
  --variant-reference data/experiments/mug0_local_mcp_20261106_163701/best_reference/reference.json \
  --output "$l20_mcp_cross"
cat "$l20_mcp_cross/REPORT.md"
cat "$l20_mcp_cross/reproducibility.json"
xdg-open "$l20_mcp_cross/zero_a/contacts/index.html"
```

实际未运行，CPU计划检查通过。关注两个进程均6/6且process_reproducible=true；全通过时细化准入false是无失败可修复，不是验证失败。

## 组18已通过：冻结MCP候选，在五个新平移对照原参考

`mug0_mcp_cross18_20261106_180627` 两次均6/6、完整轨迹复现，保留了组18原成功能力。
下一轮固定参数，用seed20261113五个新平移，原参考与改良参考各双独立进程，不训练：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_ref_confirm="data/experiments/mug0_mcp_reference_confirm5_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/compare_isaacsim_references.py \
  --variant-reference data/experiments/mug0_local_mcp_20261106_163701/best_reference/reference.json \
  --episodes 5 --position-noise 0.0005 --seed 20261113 \
  --output "$l20_ref_confirm"

cat "$l20_ref_confirm/REPORT.md"
xdg-open "$l20_ref_confirm/index.html"
```

共20个新进程，每参考30次方向试验、独立平移5个；双重复只验证复现，不增加样本数。
约25分钟为此前耗时估计，已有任务会排队。先看refined_only修复、original_only损害和完整六方向，再看连续误差。
本工具会重新测试原参考，不使用旧同进程序列的标签。46项CPU测试与实际plan-only检查通过，实际新样本仿真未运行。

## 新五样本确认未满足晋升标准：先单独食指消融

`mug0_mcp_reference_confirm5_20261106_184726` 原/联合参考均21/30，修复3也损害3，完整六方向0/5→2/5，但组4从4/6降为1/6。
联合候选不作为正式改善参考，不进入RRL。下一步仅保留食指MCP+0.015rad，去掉中指修改，在最差组4诊断贡献，不调新幅度：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_index_only="data/experiments/mug0_index_only_group4_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/refine_isaacsim_reference_local.py \
  --audit-results data/experiments/mug0_mcp_reference_confirm5_20261106_184726/group_004/original \
  --joint-deltas index_mcp_pitch=0.015 \
  --output "$l20_index_only"
cat "$l20_index_only/REPORT.md"
cat "$l20_index_only/results.json"
.venv_l20/bin/python scripts/plot_isolated_contacts.py --run "$l20_index_only/coupled_joints/a"
xdg-open "$l20_index_only/coupled_joints/a/contacts/index.html"
```

目录名coupled_joints是通用候选名，此命令只改一个食指关节。计划检查通过，实际未运行。
即使恢复4/6也只是消除损害，不等于修复原失败或取得泛化改善。
