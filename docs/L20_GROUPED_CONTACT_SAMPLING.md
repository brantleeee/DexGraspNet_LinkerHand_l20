# L20五指分组接触采样（2026-11-18）

当前状态（2026-11-18）：**五指分槽方案也已撤回**。用户确认恢复其修改之前的4个全局槽，见[当前恢复版本及命令](L20_RESTORED_GLOBAL4.md)。下文为历史记录，`--contact-sampling`及proximity相关参数已不适用于当前入口。

本次只改变接触槽分配。活动生成入口继续使用恢复后的 `utils.hand_model.HandModel` 凸包穿透查询；未启用原网格或 verified 实验模型。

## 行为

- `main.py`、`generate_small_grasps.py`、`generate_l20_v6_drill.py` 默认 `--contact-sampling grouped5`。
- 五槽顺序固定：拇指、食指、中指、无名指、小指，各从本指全部候选池采一个点。
- 初始化与优化器随机换点共享 `sample_contacts`。换点不能跨手指，未切换槽保持不变；五个池互斥，因此同一构型内不会重复选同一点。
- 缺少任一手指候选、错误槽数或已有点不属于对应手指时报错，不退回全局采样。
- 退火接受与拒绝均保留上述约束。可再次采中该槽自己的旧点，这不构成槽间重复。
- 原v6的403点/法向、12根部豁免、mimic与角限、能量公式及权重保持。接触数由4变5会改变能量数值，不直接用旧总能量判断改善。

原始4全局槽仅作显式历史对照：`--contact-sampling global`。此模式保留旧的可重复、可遗漏手指行为。直接调用main时，grouped5只能配 `--n_contact 5`；global默认4。

每个NPY候选保存 `contact_sampling`、`n_contact`、初始 `contact_point_indices_st` 和最终 `contact_point_indices`。显式接触库运行还在 `generation_contact_model.json` 保存采样模式、库身份及初始化/换点/模型源码散列。

## 验证证据

12项测试通过：分组与唯一性、全换/不换/混合换点、候选池缺失与非法输入、真实L20混合接受/拒绝后的FK/法向/梯度、原mimic和根部规则。

`data/experiments/l20_grouped5_convex_smoke_20261118` 完成2候选、2迭代GPU及Canvas导出。初始和最终均五指各一、不重复；具体记录见 `sampling_validation.json`。短检静态阈值0/2，只证明运行链路，不证明稳定抓取。

## 电钻6000步命令

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

l20_run="l20_v6_grouped5_drill_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/generate_l20_v6_drill.py \
  --name "$l20_run" \
  --contact-sampling grouped5 \
  --batch-size 32 --n-iter 6000 --top-k 16

.venv_l20/bin/python scripts/open_l20_grasp_results.py \
  --run "data/experiments/$l20_run"
```

使用清理电钻原尺寸scale1，v6接触库、法向权重1和all-roots-angle-only沿用专用入口默认值。结果位于 `results/scales/scale_1/top_16_canvas/index.html`，支持无WebGL浏览器。排名前16不表示16个均通过几何筛查。

五指都有优化槽只排除了“没有分配中指目标”这一机制；不强制中指三个link全部贴合，也不证明真实接触力、无穿透或600g承载。下一步先查选中点距离、穿透及各指关节，再决定是否进入动力学。
