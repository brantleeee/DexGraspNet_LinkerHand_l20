# 指定model_clean.obj重新生成ShadowHand抓取

2026-11-15，使用本地GraspQP与其自带ShadowHand，沿用进程内运动学兼容修正。只生成几何候选及静态HTML，不执行Isaac Sim。

源文件：`/home/lee/robot_learning/pose_estimation/rm_arm_mujoco/assets/objects/power_drill/grasp_clean/model_clean.obj`。

- 5720顶点、11264三角面；尺寸255.5×223.4×75mm。
- 比例1，保持输入坐标、朝向与偏移，不再次归一化/居中；输入OBJ逐字节复制，SHA256为47d5fba7868be887e7b58003bc2ccd9372a89e173c970ccb25039af0e62b4d54。
- 该文件不是旧的闭合实体并集。其清理记录是精确位置焊接、删除退化面和未引用顶点，当前仍不封闭。采用GraspQP支持的WARP generalized winding-number符号查询；开放表面内外判别仍有近似，不代表闭合碰撞模型已准备好。
- 同目录语义标注均UNKNOWN，不能声称生成器已避开开关/钻头或只抓手柄；本次仅使用几何。

新增接口：`--object-mesh`、`--mesh-scale`；`synthesize`只执行generate→export→静态render，跳过validate。默认老文件/老比例命令行为保持。未封闭网格导出记录`physics_admitted=false`、`mass_estimate_valid=false`；其体积推算值不能用作可信物理质量。validate入口会拒绝这类输入，未来验证需先指定闭合物理模型和质量惯量。

## 已执行

短检查`data/experiments/graspqp_clean_obj_smoke_20261115`：2候选×2步，2候选导出合法，FK误差4.614e-7，原文件和生成输入SHA一致。2项测试通过：原尺寸/偏移/文件内容保持、开放网格不准入动力学。

正式目录`data/experiments/graspqp_clean_obj_shadow_20261115`：32候选×3000步，选能量前16中的关节合法候选导出，越界原下标单独记录，不截断角度。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20
tr '\r' '\n' < data/experiments/graspqp_clean_obj_shadow_20261115.log | tail -n 12

# 等生成及导出结束后打开
xdg-open data/experiments/graspqp_clean_obj_shadow_20261115/generation_index.html
```

任务已启动，不需重复运行。以下用于以后独立新批次：

```bash
clean_out="data/experiments/graspqp_clean_obj_$(date +%Y%m%d_%H%M%S)"
.venv_l20/bin/python scripts/graspqp_drill_pipeline.py synthesize \
  --object-mesh /home/lee/robot_learning/pose_estimation/rm_arm_mujoco/assets/objects/power_drill/grasp_clean/model_clean.obj \
  --mesh-scale 1 --batch-size 32 --n-iter 3000 --top-k 16 \
  --output "$clean_out" > "$clean_out.log" 2>&1
```

生成成功只代表得到几何候选，须另外核对接触、穿透和物理执行，不能从“clean”文件名或能量排名推断稳定性。
