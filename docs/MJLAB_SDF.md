# 此前mjlab抓取环境的SDF碰撞入口

2026-11-05本机版本：mjlab1.6.0、MuJoCo3.16.0、MuJoCo-Warp3.16.0、Warp1.18.0。
这些版本支持原生网格SDF，因此对现有手部STL/物体OBJ采用 `geom type="sdf"`，无需另写每个网格的解析C++/Warp插件。
这与SDF文件格式URDF的Simulation Description Format无关。

## 已完成与边界

- 新入口 `scripts/prepare_mjlab_sdf.py` 导出独立reference，不覆盖旧凸碰撞reference。
- 22个手部collision mesh改为SDF，原visual mesh不变；物体改用一个完整来源网格的SDF，不再逐凸块碰撞。
- 固定原编译模型的质量、惯量、惯性坐标，保留关节、5个mimic等式、驱动、摩擦、初始参考与scale。
- 网格文件与XML散列存入reference；加载时检查，拒绝凸/SDF检查点交叉加载。
- 默认物体来源仍是 `coacd/decomposed.obj`，不能恢复CoACD前丢失细节。可传 `--object-mesh` 指定同坐标系、同归一化单位的更完整源网格；scale仍取reference。
- CPU三项测试通过；真实GPU单环境8步检查完成，运行时确认23个SDF、0个凸mesh碰撞、5个关节等式，观测和奖励有限。
- 这只证明接入可运行，尚未进行完整抓取稳定性或性能吞吐评估，也不证明与Isaac Sim SDF表面/求解器完全一致。

## 导出和短检查

已导出的旧杯子候选7reference：`data/mjlab_grasp/scale_0p06_candidate_7_sdf`。
注意这是旧文件的候选索引，不是当前Isaac Sim杯子0任务；来源身份保存在reference.json。

```bash
cd /home/lee/DexGraspNet_LinkerHand_l20

# 导出已有结果到一个新目录；此命令不运行仿真。
/home/lee/robot_learning/mjlab/.venv/bin/python scripts/prepare_mjlab_sdf.py \
  --reference data/mjlab_grasp/scale_0p06_candidate_7 \
  --output data/mjlab_grasp/scale_0p06_candidate_7_sdf_new

# 对已导出的SDF reference短检查，不训练。
bash scripts/mjlab_grasp.sh \
  --mode smoke --task-version v3 \
  --reference data/mjlab_grasp/scale_0p06_candidate_7_sdf \
  --num-envs 1 --steps 8 \
  --output "data/experiments/mjlab_sdf_check_$(date +%Y%m%d_%H%M%S)"
```

上面选择旧mjlab v3保留静态拇指参考；该版本号与原生Isaac Sim残差v3/v4无关。
旧mjlab v4/v5会释放静态拇指依赖，不能将其零动作理解成原始全部静态角度。
首次SDF内核编译较慢。稳定性对照应固定此任务版本及控制，仅改变reference的碰撞表示；SDF训练须新建策略。
原独立refinement入口尚未增加SDF导出开关；此入口作用于旧RL任务的 `--reference`，不能宣称所有旧验证器已切换。

验证结果：`data/experiments/mjlab_sdf_audit_smoke_20261105/sdf_runtime_audit.json`、`smoke.json`。

官方支持：[MJWarp SDF](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#sdf-plugins)。
