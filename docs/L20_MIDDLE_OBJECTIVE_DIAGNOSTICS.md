# L20中指目标函数诊断：运行说明

2026-11-16。实现[对照方案](L20_MIDDLE_ABLATION_PLAN.md)的固定姿态机理检查；不是四组重新生成，也不是Isaac Sim验证。原生成器、模型、旧抓取和RRL检查点均不改动。

## 完整诊断命令

需要本工程`.venv_l20`和CUDA，使用原TorchSDF能量/反向传播。候选下标是NPY下标，22/18是当前排名前两位，31是中指MCP已弯曲的对照。比例直接读取保存值1，不缩放手腕。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

l20_diag_out="data/experiments/l20_middle_objective_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/diagnose_l20_middle_objective.py \
  --result-file data/experiments/l20_clean_drill_20261115/results/scales/scale_1/power-drill-clean-l20.npy \
  --object-code power-drill-clean-l20 \
  --candidates 22 18 31 \
  --device cuda:0 --seed 1 \
  --repeats 8 --surface-samples 2000 \
  --scan-deltas 0 0.015 0.05 0.10 \
  --fd-epsilon 0.001 --geometry-pairs 8 \
  --output "$l20_diag_out"

cat "$l20_diag_out/REPORT.md"
xdg-open "$l20_diag_out/index.html"
```

输出目录必须为新目录；脚本不会覆盖旧输出。没有训练轮数，没有抓取优化，也不修改候选。每种接触选择有8组四点探针；扫描只复用每种模式的第0组，不将不同接触点的损失当作同一条曲线。

## 实际检查内容

- 自排斥按同连杆、直接相邻、同指非相邻、指掌、不同指分解损失和独立关节梯度，列出中指相关连杆对的加权MCP/PIP梯度。
- 同一冻结姿态下比较global4、without_middle4、grouped4。最后一项槽位为拇指、食指、中指、无名指/小指联合池，总点数都为4，保存共同均匀随机数及实际索引。
- 每种接触集合分别用原排斥和屏蔽同体/直接相邻的诊断掩码计算，掩码仅用于消融，不修改正式碰撞规则。
- 梯度先按mimic链式法则回传；原始能量与原生成器`cal_energy`的数值、梯度直接对照，若不匹配则报错。
- 固定其他关节与手腕，扫描中指MCP增量，记录各指接近距离、原符号距离内部值和能量；另用有限差分核对MCP/PIP自排斥导数，边界使用投影后的单边差分。
- 排斥贡献靠前的连杆对做双向顶点采样到真实三角网格的距离检查。若目标手部网格开放或绕向不一致，不输出内部深度，而不是用不可靠符号宣称穿透。

## 输出及阅读顺序

1. `REPORT.md`：各类别梯度及接触探针均值。正梯度的下降方向为减小该角度。
2. `pair_gradients.csv`：先看推动MCP张开的具体连杆对，以及掌部或其他项是否抵消。
3. `contact_probes.csv`：比较不含中指和包含中指后，接触距离梯度、总梯度及限位投影的变化。
4. `index.html`：MCP扫描能量曲线、接近距离曲线及各候选的三维探针页面。
5. `results.json`：完整分项梯度、连杆对采样几何、有限差分、源qpos和实际评估姿态。
6. `contact_sets.json`、`object_surface_points.npy`、`config.json`：明确的新接触集合、完整模式下固定的未缩放物体表面采样、参数/源文件/网格散列。`status.json`区分完整、部分模式和错误。

旧NPY没有最终接触索引，故这不是旧接触记录的重建。投影更新使用单位预条件，不能解释为旧RMSProp的真实一步。物体开放网格仍沿用原生成器的距离符号；原E_pen是采样深度之和，不是物理验证器的最大穿透。连杆顶点采样没有检出内部点也不能证明连续网格不相交。

## 无GPU时的部分检查

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

.venv_l20/bin/python scripts/diagnose_l20_middle_objective.py \
  --device cpu --self-only --candidates 22 18 31 \
  --output "data/experiments/l20_middle_self_only_$(date +%Y%m%d_%H%M%S)"
```

此模式仅分解自排斥及有限差分，不检查接触吸引和E_pen；不能据此下完整目标函数结论。脚本不会在CUDA不可用时静默降级。

## 已完成的短检查与新线索

- 3项CPU测试通过：分组接触覆盖/重现、点对互斥完备及非相邻保留、刚体内部排斥旋转不变而跨刚体可产生梯度。
- CPU候选22部分检查完成；CUDA候选22和31、各模式2组探针、64物体采样点、MCP增量0和.015、前2连杆对采样及自包含HTML输出完成，目录`l20_middle_objective_cuda_checked_20261116`。这是短接口检查，默认2000点/8组/3候选命令尚未代用户运行。
- 候选22的自排斥加权MCP梯度：食指近节—中指近节约+0.7673，中指近节—无名指近节约+0.2923，掌部—中指近节约−0.5349，净值约+0.5247。同体及直接相邻贡献接近零。因此仅去掉同体/相邻项，并未改变该MCP的主要张开趋势；对PIP的作用须另看，不能外推全部关节。
- 一组受控探针中，不含中指时其距离项MCP梯度为0，包含中指后出现闭合梯度。该短检查使用64个物体采样点，且接触点是新探针，不能当成旧候选22完整失败原因已确认。

下一步先运行完整诊断，检查上述关系能否在候选18、31及更多接触探针中保持，再决定是否开展分组接触生成，以及如何处理跨指20mm排斥。实际手部网格也存在开放面，真实相交判定仍需可靠几何/物理复核；不按原始符号值盲删避碰约束。
