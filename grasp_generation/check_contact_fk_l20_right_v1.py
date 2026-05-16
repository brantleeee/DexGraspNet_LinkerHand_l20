"""
L20 接触点、网格与本体局部坐标系、掌心对齐可视化验证脚本
"""
import os
import sys
import torch
import numpy as np
import plotly.graph_objects as go

# 稳定切换到 grasp_generation 根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from utils.hand_model import HandModel

device = 'cpu'

print("🔧 [L20 右手] 正在从 l20_right_description 加载 HandModel...")

# 适配最新的目录结构
hand_model = HandModel(
    model_path=os.path.join(ROOT_DIR, 'l20_right_description/linkerhand_l20_right.urdf'),
    mesh_path=os.path.join(ROOT_DIR, 'l20_right_description/meshes'),
    contact_points_path=os.path.join(ROOT_DIR, 'l20_right_description/contact_points.json'),
    penetration_points_path=os.path.join(ROOT_DIR, 'l20_right_description/penetration_points.json'),
    n_surface_points=0,
    device=device,
    force_mesh_sdf=True
)

n_dofs = hand_model.n_dofs
# 初始化手部姿态：位置(3) + 旋转Ortho6d(6) + 关节(n_dofs)
hand_pose = torch.zeros((1, 9 + n_dofs), dtype=torch.float, device=device)
hand_pose[0, 3] = 1.0  # Ortho6D x-axis
hand_pose[0, 7] = 1.0  # Ortho6D y-axis

# 可以设置关节全 0，也可以设置微曲
# hand_pose[0, 9:] = 0.5 * (hand_model.joints_lower + hand_model.joints_upper)

hand_model.set_parameters(hand_pose)

fig = go.Figure()
total_contact = 0

print("\n🚀 正在进行空间变换与渲染...")
print("-" * 50)

# ================= 👑 局部坐标轴基准点 =================
AXIS_LEN = 0.02  # 局部坐标轴长度 2cm
local_axes_base = torch.tensor([
    [0.0, 0.0, 0.0],
    [AXIS_LEN, 0.0, 0.0],
    [0.0, AXIS_LEN, 0.0],
    [0.0, 0.0, AXIS_LEN]
], dtype=torch.float, device=device)

# ================= 🎯 掌心诊断基准点 =================
# 这里填入你在 initializations.py 中设置的掌心偏移量
palm_offset_local = torch.tensor([0.0, 0.0, 0.08], dtype=torch.float, device=device)

for link_name in hand_model.mesh:
    # 注意：HandModel 解析出的键值是 'vertices' 和 'faces'
    has_mesh = 'vertices' in hand_model.mesh[link_name] and len(hand_model.mesh[link_name]['vertices']) > 0
    has_contact = 'contact_candidates' in hand_model.mesh[link_name] and len(hand_model.mesh[link_name]['contact_candidates']) > 0

    # 1. 渲染半透明骨架
    if has_mesh:
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

        # 2. 渲染该连杆的局部坐标系
        axes_transformed = hand_model.current_status[link_name].transform_points(local_axes_base)
        if len(axes_transformed.shape) == 3: axes_transformed = axes_transformed[0]
        axes_transformed = axes_transformed @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
        axes_transformed = axes_transformed.detach().cpu().numpy()

        origin = axes_transformed[0]
        pt_x, pt_y, pt_z = axes_transformed[1], axes_transformed[2], axes_transformed[3]

        fig.add_trace(go.Scatter3d(x=[origin[0], pt_x[0]], y=[origin[1], pt_x[1]], z=[origin[2], pt_x[2]], mode='lines', line=dict(color='red', width=4), name=f"{link_name} +X", hoverinfo='name', showlegend=False))
        fig.add_trace(go.Scatter3d(x=[origin[0], pt_y[0]], y=[origin[1], pt_y[1]], z=[origin[2], pt_y[2]], mode='lines', line=dict(color='green', width=4), name=f"{link_name} +Y", hoverinfo='name', showlegend=False))
        fig.add_trace(go.Scatter3d(x=[origin[0], pt_z[0]], y=[origin[1], pt_z[1]], z=[origin[2], pt_z[2]], mode='lines', line=dict(color='blue', width=4), name=f"{link_name} +Z", hoverinfo='name', showlegend=False))

    # 3. 渲染接触点 (红点)
    if has_contact:
        pts = hand_model.mesh[link_name]['contact_candidates']
        n_pts = pts.shape[0]
        total_contact += n_pts
        
        pts_transformed = hand_model.current_status[link_name].transform_points(pts)
        if len(pts_transformed.shape) == 3: pts_transformed = pts_transformed[0]
        pts_transformed = pts_transformed @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
        pts_transformed = pts_transformed.detach().cpu().numpy()

        fig.add_trace(go.Scatter3d(
            x=pts_transformed[:, 0], y=pts_transformed[:, 1], z=pts_transformed[:, 2],
            mode='markers', marker=dict(size=4, color='red', opacity=1.0),
            name=f"{link_name} (接触点)"
        ))
        print(f"✅ {link_name.ljust(25)} : 渲染了 {n_pts} 个接触点")
    elif has_mesh:
        print(f"➖ {link_name.ljust(25)} : 无主动接触点")

# ================= 🎯 渲染掌心诊断射线 =================
root_link_name = list(hand_model.mesh.keys())[0]
base_matrix = hand_model.current_status[root_link_name].get_matrix()[0]

# 计算虚拟掌心位置
palm_center = (base_matrix[:3, :3] @ palm_offset_local + base_matrix[:3, 3])
palm_center = palm_center @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
palm_center = palm_center.detach().cpu().numpy()

# 计算手腕位置
base_pos = (base_matrix[:3, 3] @ hand_model.global_rotation[0].T + hand_model.global_translation[0]).detach().cpu().numpy()

fig.add_trace(go.Scatter3d(
    x=[palm_center[0]], y=[palm_center[1]], z=[palm_center[2]],
    mode='markers', marker=dict(color='magenta', size=8, symbol='diamond'),
    name='Virtual Palm Center (虚拟掌心)'
))

fig.add_trace(go.Scatter3d(
    x=[base_pos[0], palm_center[0]], y=[base_pos[1], palm_center[1]], z=[base_pos[2], palm_center[2]],
    mode='lines', line=dict(color='magenta', width=6, dash='dash'), 
    name='Wrist to Palm (手腕至掌心连线)'
))

print("-" * 50)
# 全局坐标轴
fig.add_trace(go.Scatter3d(x=[0, 0.1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=8), name='Global +X'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0.1], z=[0, 0], mode='lines', line=dict(color='green', width=8), name='Global +Y'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 0.1], mode='lines', line=dict(color='blue', width=8), name='Global +Z'))

fig.update_layout(
    title=f"L20 接触点与局部坐标轴 - 共 {total_contact} 🔴",
    scene=dict(aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=40)
)

output_html = os.path.join(ROOT_DIR, "l20_right_contact_fk_vis.html")
fig.write_html(output_html)
print(f"🎉 网页已保存至: {output_html}")
print("💡 提示: 品红色的菱形代表【算法认为的掌心】。请调整 palm_offset_local 让它卡在物理手掌的中心。")