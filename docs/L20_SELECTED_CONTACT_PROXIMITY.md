# 五指分槽后仍悬空：诊断与接近目标修正

> **已撤回（2026-11-18）**：用户反馈程序仍不正确，已恢复到本次修改前的grouped5版本。下文为历史实验记录，proximity相关参数和局部优化命令不再适用于当前入口；新增实验模块/测试/优化脚本已归档到 `data/code_backups/pre_revert_proximity_20261118_161719` 的 `.snapshot` 文件，结果数据保留。当前流程见 [五指分组采样](L20_GROUPED_CONTACT_SAMPLING.md)。

日期：2026-11-18。范围：静态抓取生成，不是RRL或动力学验证。

## 现象与证据

一次性读取已完成的 `l20_v6_grouped5_drill_20261118_143942`，32候选、6000步、scale1、v6库403点。按真实 `HandModel.global_index_to_link_index` 映射，32/32均覆盖五指，不能继续把这轮问题归因为中指没分槽。

CPU原OBJ三角面最近距离表明：所选中指点≤1mm仅9/32，中指任意点≤1mm为24/32；前16分别8/16和12/16。不能用“任意一点很近”替“实际选中点合格”。审计见 `data/experiments/grouped5_middle_audit_20261118/REPORT.md`。

|例子|证据|解释|
|---|---|---|
|候选27|所选点3.706mm，其他中指点最近0.005mm|已有近点，但离散选择未选中|
|候选22（能量第3）|所选24.023mm，整指最近22.585mm，E_fc仅0.032|整指悬空仍可因软目标权衡得到较低能量|
|候选18/8/29|所选近节对PIP的FK Jacobian严格0|这个槽无法提供PIP闭合驱动|
|候选1/4|所选原面距离约0.318/0.292mm|不是所有结果的中指都悬空，不应统一强迫增加角度|

冻结梯度探针中，候选22 MCP增加0.1rad后距离项梯度−0.446（促闭合），但自排斥+2.176抵消，合计+1.720仍促张开。该探针采用新固定2000点采样，不能当原始E_pen的精确历史重放。见 `grouped5_middle_objective_audit_20261118`。MCP弯曲也不是距离的单调函数，不能简单固定一个更大角度。

## 代码行为变化

新模式 `--contact-objective proximity` 为三个生成入口的默认值，要求 `grouped5`；真正旧版对照使用 `--contact-objective legacy`。

1. **运动链有效池**：普通四指使用middle/distal，拇指使用proximal/distal，避免掌骨或普通近节槽无法驱动PIP。所有点仍来自当前版本化接触库，没有重新制造表面点。连杆筛选不等于已认证“指腹区域”。
2. **固定姿态内重选**：初始化后和每次姿态提案后，在各指池中按原物体查询的距离绝对值选最近点。其他指不能占用该槽，槽间仍唯一；不允许远点仅因E_fc低而胜出。
3. **逐点超距目标**：保留原E_dis、E_fc和其他项，额外加入

   `E_contact = sum(max(abs(d_i) - tolerance, 0)^2 / tolerance)`。

   默认tolerance=0.001m，w_contact=100。它惩罚某一根手指的大误差，而不是让几根手指的小距离掩盖悬空。原始手物穿透、自排斥及12根部规则未删除。
4. **提案护栏**：新模式对既有采样E_pen增加接受约束：原来≤0.001的构型继续保持≤0.001；初始已越界则新提案不能使该值增加。这不意味着初始越界自动消失，也不是完整网格防穿透保证。
5. **显式拒绝悬空结果**：保存实际所选五点的距离与连杆。每点均≤1mm才能 `contact_admitted=True`；再结合原静态E_fc/E_dis/E_pen阈值决定 `geometry_admitted`。正式排名只列合格项；没有合格项就输出0个，绝不自动把失败姿态当成成功。`diagnostic_top_K_canvas`保留未过滤的诊断，标题标FAIL。

新增模块为 `grasp_generation/utils/contact_proximity.py`；共享优化器仍负责原mimic投影和接受/拒绝恢复，只增加selector回调和proposal_allowed过滤。原 `hand_model.py`、`energy.py`、`root_repulsion.py` 及根部策略源SHA与旧批次一致。没有启用已撤回的raw/verified穿透分支。

保存 `object_surface_points.npy`，便于以后固定同一采样复算；每个候选保存E_contact、权重、选中点距、初始E_pen（本次补充）、contact/geometry准入。新目标总能量不能与旧目标总能量直接比较。换参考后不可直接沿用旧RRL检查点。

## 已验证与尚未验证

短局部对照：同初始身体姿态、固定物体采样，仅允许中指MCP/PIP及mimic DIP变化，100步；两组使用相同采样穿透非增护栏。新模式同时包含重选与新损失，不声称分离其贡献。

|候选|旧目标初始→最终mm|新目标初始→最终mm|结论|
|---|---:|---:|---|
|22|24.023→24.068|22.585→22.528|仍失败，不能强行放行|
|27|3.707→3.707|0.100→0.122|主要是初始重选收益，不是迭代修复|
|18|14.659→15.081|24.315→4.430|产生闭合与接近响应，仍未达1mm|

三者E_pen没有增加，mimic误差<5×10⁻⁸rad。TorchSDF原距离带约0.1mm正则底值，因此这里的小距离不等于CPU原面距离，更不等于真实接触力。见 `data/experiments/grouped5_contact_refinement_short_20261118/REPORT.md`。

完整入口短检 `l20_proximity_smoke100_20261118`：4候选100步完成，NPY/能量重组/分指身份/Canvas导出正确；0/4通过接近与静态代理筛查，2个初始穿透的构型结束仍越界。因此只证明链路与拒绝机制，不证明新生成已经收敛或解决所有中指失败。诊断页包含全部4个FAIL；正式排名0个是正确行为。`proximity_validation.json`保留原始数值。此运行结束后又补充了初始E_pen字段及元数据描述；不改写其旧源码散列或声称它已包含补充字段。

## 操作与下一步

先复查已完成的局部对照与诊断：

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
cat data/experiments/grouped5_contact_refinement_short_20261118/REPORT.md
xdg-open data/experiments/l20_proximity_smoke100_20261118/results/scales/scale_1/diagnostic_top_4_canvas/index.html
```

如需重跑受控局部对照，脚本将迭代数限制为≤200，避免误开长任务：

```bash
.venv_l20/bin/python scripts/check_contact_proximity_refinement.py \
  --run data/experiments/l20_v6_grouped5_drill_20261118_143942 \
  --candidates 22 27 18 --iterations 100 \
  --output "data/experiments/contact_refinement_check_$(date +%Y%m%d_%H%M%S)"
```

新生成入口可显式设置 `--contact-objective proximity --contact-tolerance 0.001 --w-contact 100`。`--w-contact 0`仍会改变选点和准入，不是旧版对照；旧模式必须显式 `--contact-objective legacy`。本轮未启动6000步。

下一步先检查候选22的可达性和局部极值：固定手腕时，MCP/PIP是否存在同时满足近接与碰撞护栏的构型；若没有，再分析手腕自由度，而不是反复增大闭合角或关闭自碰撞。得到合格静态构型后还要查绝对几何、闭合路径和600g动力学。几何接近、采样穿透代理合格、物理稳定是三层不同的结论。
