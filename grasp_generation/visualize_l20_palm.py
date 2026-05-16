"""
LinkerHand L20 掌心与抓取方向可视化诊断脚本
"""
import os
import sys
import torch
import numpy as np
import plotly.graph_objects as go
import math
import transforms3d

# 稳定切换到 grasp_generation 根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from utils.hand_model import HandModel

device = 'cpu'

print("🔧 正在加载 LinkerHand L20 Model...")

# 适配 L20 的目录结构
hand_model = HandModel(
    model_path=os.path.join(ROOT_DIR, 'l20_right_description/linkerhand_l20_right.urdf'),
    mesh_path=os.path.join(ROOT_DIR, 'l20_right_description/meshes'),
    contact_points_path=os.path.join(ROOT_DIR, 'l20_right_description/contact_points.json'),
    penetration_points_path=os.path.join(ROOT_DIR, 'l20_right_description/penetration_points.json'),
    n_surface_points=0,
    device=device,
    force_mesh_sdf=True
)

# 初始化手部姿态
n_dofs = hand_model.n_dofs
hand_pose = torch.zeros((1, 9 + n_dofs), dtype=torch.float, device=device)
hand_pose[0, 3] = 1.0  # Ortho6D x-axis
hand_pose[0, 7] = 1.0  # Ortho6D y-axis

# 让 L20 保持所有关节为 0 的自然伸直姿态
hand_pose[0, 9:] = torch.zeros_like(hand_model.joints_lower)

hand_model.set_parameters(hand_pose)

fig = go.Figure()

print("\n🚀 正在渲染 L20 网格与局部坐标系...")
print("-" * 50)

# ================= 👑 局部坐标轴基准点 =================
AXIS_LEN = 0.02  # 局部坐标轴长度 2cm
local_axes_base = torch.tensor([
    [0.0, 0.0, 0.0],
    [AXIS_LEN, 0.0, 0.0],
    [0.0, AXIS_LEN, 0.0],
    [0.0, 0.0, AXIS_LEN]
], dtype=torch.float, device=device)

# ================= 🎯 掌心诊断基准点 (核心调参区) =================
# 💡 这里填入你在 initializations.py 中设置的掌心偏移量。
# 比如假设真正的掌心在手腕 (base_link) 正前方 8cm 处：
palm_offset_local = torch.tensor([0.0, 0.0, 0.03], dtype=torch.float, device=device)

for link_name in hand_model.mesh:
    if 'vertices' not in hand_model.mesh[link_name] or len(hand_model.mesh[link_name]['vertices']) == 0:
        continue
        
    # 渲染半透明骨架
    v = hand_model.current_status[link_name].transform_points(hand_model.mesh[link_name]['vertices'])
    if len(v.shape) == 3: v = v[0]
    v = v @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
    v = v.detach().cpu().numpy()
    f = hand_model.mesh[link_name]['faces'].detach().cpu().numpy()
    
    fig.add_trace(go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], 
        i=f[:, 0], j=f[:, 1], k=f[:, 2], 
        color='lightgray', opacity=0.4, 
        name=link_name, hovertemplate='%{name}'
    ))

    # 渲染该连杆的局部坐标系
    axes_transformed = hand_model.current_status[link_name].transform_points(local_axes_base)
    if len(axes_transformed.shape) == 3: axes_transformed = axes_transformed[0]
    axes_transformed = axes_transformed @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
    axes_transformed = axes_transformed.detach().cpu().numpy()

    origin = axes_transformed[0]
    pt_x, pt_y, pt_z = axes_transformed[1], axes_transformed[2], axes_transformed[3]

    fig.add_trace(go.Scatter3d(x=[origin[0], pt_x[0]], y=[origin[1], pt_x[1]], z=[origin[2], pt_x[2]], mode='lines', line=dict(color='red', width=4), name=f"{link_name} +X", hoverinfo='name', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[origin[0], pt_y[0]], y=[origin[1], pt_y[1]], z=[origin[2], pt_y[2]], mode='lines', line=dict(color='green', width=4), name=f"{link_name} +Y", hoverinfo='name', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[origin[0], pt_z[0]], y=[origin[1], pt_z[1]], z=[origin[2], pt_z[2]], mode='lines', line=dict(color='blue', width=4), name=f"{link_name} +Z", hoverinfo='name', showlegend=False))

# ================= 🎯 渲染掌心诊断射线与接近方向 =================
root_link_name = list(hand_model.mesh.keys())[0]  # 对于 L20 通常是 base_link
base_matrix = hand_model.current_status[root_link_name].get_matrix()[0]

# 1. 计算虚拟掌心位置
palm_center = (base_matrix[:3, :3] @ palm_offset_local + base_matrix[:3, 3])
palm_center = palm_center @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
palm_center = palm_center.detach().cpu().numpy()

# 计算手腕位置 (画线用)
base_pos = (base_matrix[:3, 3] @ hand_model.global_rotation[0].T + hand_model.global_translation[0]).detach().cpu().numpy()

# 2. 💡 计算 L20 的算法接近射线 (核心逻辑)
# 在 initializations.py 中，L20 的姿态矩阵是：R_final = R_global @ R_local @ rotation_shadow @ linker_fix
# 这意味着基座相较于“标准抓取姿态”，经历了 rotation_shadow 和 linker_fix 两次旋转。
# 所以我们要寻找物体在手掌局部坐标系下的方向，必须使用这俩矩阵乘积的转置 (.T)

# Shadow Hand 固有的 -60 度旋转
R_shadow = transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz')
# L20 专属的 Z 轴 -90 度修正
R_linker_fix = transforms3d.euler.euler2mat(0, 0, math.radians(-90.0), axes='sxyz')

# 总旋转矩阵
R_hand_total = torch.tensor(R_shadow @ R_linker_fix, dtype=torch.float, device=device)

# 使用转置矩阵 .T 逆向寻找 Z 轴 (即指向物体的方向)
approach_vec_local = R_hand_total.T @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float, device=device)

# 将射线从局部空间变换到全局空间
approach_vec_global = base_matrix[:3, :3] @ approach_vec_local
approach_vec_global = approach_vec_global @ hand_model.global_rotation[0].T
approach_vec_global = approach_vec_global.detach().cpu().numpy()

# 延长射线以便观察 (假设拉长 15 厘米)
ray_end_pos = palm_center + approach_vec_global * 0.15

# ================= 🎨 绘制诊断元素 =================
# 画出手腕到掌心的连线
fig.add_trace(go.Scatter3d(
    x=[base_pos[0], palm_center[0]], 
    y=[base_pos[1], palm_center[1]], 
    z=[base_pos[2], palm_center[2]],
    mode='lines', line=dict(color='orange', width=4, dash='dot'), 
    name='Wrist to Palm Offset (手腕至掌心偏移)'
))

# 画出虚拟掌心（品红色大菱形）
fig.add_trace(go.Scatter3d(
    x=[palm_center[0]], y=[palm_center[1]], z=[palm_center[2]],
    mode='markers', marker=dict(color='magenta', size=8, symbol='diamond'),
    name='Virtual Palm Center (算法掌心基准点)'
))

# 画出接近射线
fig.add_trace(go.Scatter3d(
    x=[palm_center[0], ray_end_pos[0]], 
    y=[palm_center[1], ray_end_pos[1]], 
    z=[palm_center[2], ray_end_pos[2]],
    mode='lines', line=dict(color='magenta', width=6, dash='dash'), 
    name='Algorithm Approach Ray (算法接近射线)'
))

print("-" * 50)
# 全局坐标轴
fig.add_trace(go.Scatter3d(x=[0, 0.1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=8), name='Global +X'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0.1], z=[0, 0], mode='lines', line=dict(color='green', width=8), name='Global +Y'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 0.1], mode='lines', line=dict(color='blue', width=8), name='Global +Z'))

fig.update_layout(
    title=f"L20 掌心基准点与抓取方向诊断",
    scene=dict(aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=40)
)

output_html = os.path.join(ROOT_DIR, "l20_palm_vis.html")
fig.write_html(output_html)
print(f"🎉 网页已保存至: {output_html}")
print("💡 诊断说明:")
print("1. 观察品红色菱形：修改代码中的 palm_offset_local，让它精准嵌入金属手掌表面中心。")
print("2. 观察品红虚线：这根虚线应该【垂直于手掌心】射向前方空间。这就是算法起手时的对准方向！")