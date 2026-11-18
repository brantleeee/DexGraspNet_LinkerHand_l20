> **2026-11-18撤回**：用户实际长生成发现手物相互穿透。原始开放网格+TorchSDF符号不能作为可靠穿透约束，包围盒外保护不足。下文raw-mesh长生成命令不再推荐。当前已按用户确认恢复原HandModel凸包查询和4个全局槽，grouped5也已撤回；当前命令见[恢复版说明](L20_RESTORED_GLOBAL4.md)。凸包仅是保守基线，不是最终正确方案；后续需统一封闭非凸代理和独立双向几何验收，再重新开放raw路径。

# L20生成v2：原非凸查询与分指接触（2026-11-17）

**2026-11-18纠正：暂停推荐本页长生成命令。实际6000步批次0/32通过静态阈值，同源封闭物体代理的反向检查检出厘米级手部穿入。原网格查询/五指槽机制接通不等于可靠防穿透，前轮短检查验收不足。请先看[穿透失败复盘](L20_V2_PENETRATION_FAILURE_20261118.md)。本页保留为历史实现记录，不代表当前可接受的抓取算法。**

已实现两项修正，未代启动6000步长生成。现有任务不停止、不覆盖旧结果。

## 原非凸网格穿透查询

新增`utils/hand_model_v2.py`，保留旧`hand_model.py`及接触库绑定的原资产SHA。新HandModel中`force_mesh_sdf=True`实际选择raw-mesh，`face_verts`直接由原顶点/原三角面构建；不以凸包代替查询。继承初始化暂时仍构建旧凸包缓存以保持旧模型接口，但raw模式计算E_pen的实际查询缓存已替换，逐link面数输出到`penetration_geometry.json`核查。

main新增`--penetration-geometry raw-mesh|legacy-convex`，通用入口默认legacy-convex保证旧调用语义；专用电钻driver现在默认raw-mesh。不要将“默认保留旧入口”误认为新任务未接入：应核对本轮metadata和查询面数。

**处理了新发现的符号问题**：第一次2步短检原网格E_pen异常增大，发现开放面伪法向可将远场点判为内部。v2强制原始link包围盒外点为外侧，不再产生该link正穿透；没有填洞、重新缩放或替换碰撞实体。包围盒内仍沿用TorchSDF符号，开放/多壳网格仍未认证，不能承诺所有符号正确。

## 初始化与重采样使用同一分组采样器

新增`utils/contact_sampling.py`，两个调用点都使用它：`initialize_convex_hull`的初始槽，以及`Annealing.try_step`每次随机换点。

|模式|槽及含义|
|---|---|
|global|保留旧全局有放回采样，用于旧对照|
|grouped4|拇指、食指、中指、无名指/小指联合池各1槽|
|grouped5（新默认）|拇指、食指、中指、无名指、小指各1槽|

分组池互斥，检查所选ID唯一；换点只在该槽所属手指内进行，未切换的槽保持原点。拒绝提案继续恢复旧槽。grouped5自动设置n_contact=5，grouped4设置4，避免CLI槽数冲突。新选项约束手指参与，不保证选中指腹、贴合或每个link都接触；也不强制四指角度一致。

## 已执行验证

- 6项CPU测试通过：分组采样40次随机切换保留每指槽/唯一性，22link查询三角面与原网格逐项一致，以及小模型/真实L20混合接受拒绝与mimic梯度测试。
- 实际GPU 2候选、2迭代：`l20_generation_v2_bounds_smoke_20261117`，scale1、403候选、grouped5、12根部豁免、w_normal1；最终每候选5点覆盖5指且无重复，能量分项重组一致，2个Canvas2D页面已输出。静态阈值0/2，仅链路短检。
- GPU远场查询(±1,±1,±1)m均为外；已知中指中节/远节凸包填充区域在raw查询下分别约−4.888/−3.157mm，证明查询已不再沿用凸包。
- 保存原物体采样云`object_surface_points.npy`，供后续同采样梯度对照；保存algorithm_version、geometry、sampling、槽数和源散列。
- 第一次未加包围盒保护的试算目录`l20_generation_v2_smoke_20261117`仅为问题复现，不作为建议使用版本。

原法向权重和跨指20mm自排斥仍保持，E_fc仍是平衡代理，未在本轮增加力闭合或600g受力目标。原始非凸查询比凸包更耗时，旧总能量与5槽新能量不能不加说明直接排名比较。

## 新版本6000步命令

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
l20_v2="l20_v2_raw_grouped5_$(date +%Y%m%d_%H%M%S)"

.venv_l20/bin/python scripts/generate_l20_v6_drill.py \
  --name "$l20_v2" \
  --contact-library data/hand_contacts/l20_contact_v6_confirmed_20261117 \
  --penetration-geometry raw-mesh \
  --contact-sampling grouped5 \
  --root-contact-policy all-roots-angle-only \
  --batch-size 32 --n-iter 6000 --top-k 16 --w-normal 1

# 完成后叠加403候选与原物体表面接近位置：
.venv_l20/bin/python scripts/show_grasp_object_contacts.py \
  --run "data/experiments/$l20_v2" \
&& xdg-open "data/experiments/$l20_v2/results/scales/scale_1/top_16_canvas/index.html"
```

默认先32候选。需要128时显式改batch-size，先确认原网格版本耗时与显存。该命令保持清理电钻scale1（原物理尺寸），不再次缩放。

要复现旧机制，必须同时指定`--penetration-geometry legacy-convex --contact-sampling global`及原根部/法向权重。不可将旧结果元数据改写成新版本。

下一验收应核对全部候选中指槽存在、所选中指点距离/区域、真实几何穿透和六方向物理保持；槽参与本身不是稳定性证明。
