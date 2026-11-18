# 使用v6接触模型生成电钻抓取（2026-11-17）

已启动正式任务：`data/experiments/l20_v6_clean_drill_20261117_135743`，启动PID21536。此处记录启动状态，不代表任务已完成。

## 本次实际使用的模型

- L20右手原URDF/STL、原mimic关节约束。
- `data/hand_contacts/l20_contact_v6_confirmed_20261117`：403个点、17个删除稳定号、最新版Zc法向。
- 物体代码`power-drill-clean-l20`，字节相同的源文件为`/home/lee/robot_learning/pose_estimation/rm_arm_mujoco/assets/objects/power_drill/grasp_clean/model_clean.obj`。
- scale=1，原物理尺寸255.5×223.4×75mm；没有重新归一化、缩小物体或缩放手腕位姿。
- 32候选、6000次静态优化、seed=1、全局4个接触槽；初始化采用既有scale-aware分支。

原生成器只消费点坐标，不读取手部法向。单换JSON不会让前面修订的法向参与优化。本轮增加显式`--contact-library`与可选`--w-normal`，原入口不传参数时保持旧行为。新任务使用w_normal=1，这是新的实验权重，不是已证明最优值。

## 法向如何参与目标

新模块`grasp_generation/utils/contact_library.py`核对库SHA、点数量、点坐标、所属link与连续runtime映射。N按所属link的FK旋转，再按手腕旋转变换到生成世界系；**不加平移**。附加项为：

`E_normal = mean((1 + dot(hand_N_world, object_outward_N))²)`

相对两侧接触面法向应相反，故反向时为0，同向时为4。保持原E_fc（使用物体查询法向）不变，不以手部N替换物体N。

本次总目标为：`E_fc + 100 E_dis + 100 E_pen + 10 E_spen + E_joints + 1 E_normal`。

默认全手4槽仍可能不选中中指，未增加“每指必须参与”的约束；旧20mm关键点自排斥也没有在本轮替换。手物穿透项及开放电钻网格的符号近似沿用现有实现。人工方向核查、点位变更与新法向目标共同构成本次生成实验，不能把后续差异唯一归因到某一个因素。

600g质量不进入这套静态生成目标；生成完成后再用明确质量600g的物理协议验证，不能以静态能量判为承载稳定。

## 验证证据

- 2项CPU测试：反向/同向法向的代价，以及FK旋转保持梯度且不引入平移。
- GPU立方体三个已知外侧查询核对TorchSDF的返回法向符号，均与+X/+Y/+Z一致；该检查不认证开放电钻网格的全局符号。
- 实际短检查`l20_v6_clean_drill_smoke_20261117_contact403`：2候选、2迭代、scale1，403点和稳定号映射核对正确，删除点未选入，世界N单位长度和总能量分项重组成一致；保存2个Canvas2D排名页。几何阈值0/2，仅证明链路可执行。
- 正式任务已后台启动，按用户偏好不持续监控。尚无正式抓取效果或动力学改善结论。

## 重新启动一批

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
.venv_l20/bin/python scripts/generate_l20_v6_drill.py \
  --batch-size 32 --n-iter 6000 --top-k 16 --w-normal 1 --background
```

命令自动分配北京时间实验名，打印日志/状态/输出路径。没有`--background`时前台执行。输出目录或同名任务存在时拒绝覆盖。若只比较新点坐标、暂时禁用新法向项，显式使用`--w-normal 0`并作为另一实验；不可声称这种控制使用了新N优化。

## 查看本次任务

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
cat data/experiments/l20_v6_clean_drill_20261117_135743.job.json
tail -n 20 data/experiments/l20_v6_clean_drill_20261117_135743.log

# phase为complete后：
xdg-open data/experiments/l20_v6_clean_drill_20261117_135743/results/scales/scale_1/top_16_canvas/index.html
```

前16按含新法向项的总能量排名，不作稳定性排名或仅PASS筛选。Canvas2D无需WebGL；为软件显示采用0.6mm顶点聚类，**该显示简化不用于优化或碰撞**。原Plotly前16也保留于top_16。

`generation_contact_model.json`记录库/代码散列、权重与法向约定；`contact_library/`为本轮点/框架快照。每个NPY候选保存E_normal、最终选中runtime索引、稳定P号、世界N和库散列。后续应依这些证据核查中指是否入选、所选点是否接近物体，再执行物理筛选。


## 根部豁免后的新批次（2026-11-17）

最新任务为`l20_v6_roots12_drill_20261117`。在v6 403点、scale1、32候选/6000迭代、w_normal1基础上，新增`--root-contact-policy all-roots-angle-only`：生成自排斥同样关闭四指8对及拇指4对根部配对，保留其他跨指排斥、手物穿透、角限和mimic。4项测试及2候选GPU短检通过，正式任务已后台启动，结果未验收。

专用driver现在默认上述12对豁免；若要复现本文旧批次的目标，必须显式传`--root-contact-policy all`。通用main/generate_small默认仍all，不静默修改旧入口。

```bash
cat data/experiments/l20_v6_roots12_drill_20261117.job.json
# 完成后：
xdg-open data/experiments/l20_v6_roots12_drill_20261117/results/scales/scale_1/top_16_canvas/index.html
```


打开当前roots12结果建议使用单次状态检查入口，避免生成中直接打开缺失HTML：

```bash
.venv_l20/bin/python scripts/open_l20_grasp_results.py \
  --run data/experiments/l20_v6_roots12_drill_20261117
```

生成中显示进度并退出；导出完成后重新执行将打开HTML，不会持续等待或重复启动任务。
