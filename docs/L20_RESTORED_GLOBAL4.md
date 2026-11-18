# 已确认恢复：五指分槽之前的4全局槽版本

2026-11-18，用户明确确认恢复目标为“4个全局接触槽＋原凸包穿透模型”，不是grouped5。

恢复来源：`data/code_backups/pre_grouped5_20261118_143459`。六文件与备份SHA逐一相同；实际历史实验 `l20_restored_drill_20261118_141105` 记录的main、optimizer、energy、contact_library、root_repulsion及根部策略源码SHA均匹配。恢复前当前代码备份于 `data/code_backups/pre_restore_global4_20261118_180219`，含restore_verification.json。

上述代码备份及实验数据为本机溯源文件，不随GitHub代码提交上传；仓库保留恢复后的活动源码、说明和相关诊断测试。

## 当前行为

- 初始化4个全局随机槽，优化器随机换点仍从全手候选采样；允许重复，不保证中指参与。
- 原 `utils.hand_model.HandModel` 逐link凸包穿透查询、原能量公式和退火接受/拒绝逻辑。
- 电钻专用入口保留v6库403点、法向权重1、all-roots-angle-only共12对根部豁免、原角限和mimic。
- 原尺寸电钻 `power-drill-clean-l20`，scale=1。
- grouped5和proximity实验均不在活动生成路径。当前入口不接受 `--contact-sampling`、`--contact-objective`、`--contact-tolerance`、`--w-contact`。
- 分组采样器作为历史模块保留但不调用；相应实验与结果未删除。测试中撤销了已停用的grouped生产路径参数化分支，原真实手FK/梯度/拒绝恢复测试保留。

## 6000步命令

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

l20_run="l20_global4_drill_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/generate_l20_v6_drill.py \
  --name "$l20_run" \
  --contact-library data/hand_contacts/l20_contact_v6_confirmed_20261117 \
  --w-normal 1 \
  --root-contact-policy all-roots-angle-only \
  --batch-size 32 --n-iter 6000 --top-k 16
```

完成后在同一终端打开无WebGL页面：

```bash
.venv_l20/bin/python scripts/open_l20_grasp_results.py \
  --run "data/experiments/$l20_run"
```

页面为 `results/scales/scale_1/top_16_canvas/index.html`。前16是能量排名，不等于静态或动力学通过。恢复只确认版本身份，不宣称解决历史中指悬空问题。本轮不代启长生成。
