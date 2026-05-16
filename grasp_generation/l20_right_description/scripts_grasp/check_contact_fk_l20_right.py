import os
import sys
import torch
import numpy as np
import plotly.graph_objects as go

# 兼容各种路径运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hand_model import HandModel

device = 'cpu'

print("🔧 [L20 右手] 正在加载 HandModel 与主动接触点阵 FK 链条...")

hand_model = HandModel(
    urdf_path='l20_right_description/linkerhand_l20_right.urdf',
    contact_points_path='l20_right_description/contact_points.json',
    device=device
)

n_dofs = hand_model.n_dofs
hand_pose = torch.zeros((1, 9 + n_dofs), dtype=torch.float, device=device)
hand_pose[0, 3] = 1.0  # Ortho6D x-axis
hand_pose[0, 7] = 1.0  # Ortho6D y-axis

hand_model.set_parameters(hand_pose)

fig = go.Figure()
total_contact = 0

print("\n🚀 正在进行空间变换与接触点渲染 (包含局部坐标系)...")
print("-" * 50)

# ================= 👑 局部坐标轴基准点 =================
# 定义局部坐标系的 4 个关键点：原点, X轴端点, Y轴端点, Z轴端点
# 轴长度设定为 0.015 米 (1.5厘米)，避免太长互相穿模
AXIS_LEN = 0.015 
local_axes_base = torch.tensor([
    [0.0, 0.0, 0.0],
    [AXIS_LEN, 0.0, 0.0],
    [0.0, AXIS_LEN, 0.0],
    [0.0, 0.0, AXIS_LEN]
], dtype=torch.float, device=device)
# ========================================================

for link_name in hand_model.mesh:
    has_mesh = 'visual_vertices' in hand_model.mesh[link_name]
    has_contact = 'contact_candidates' in hand_model.mesh[link_name]

    # 渲染半透明骨架
    if has_mesh:
        v = hand_model.current_status[link_name].transform_points(hand_model.mesh[link_name]['visual_vertices'])
        if len(v.shape) == 3: v = v[0]
        v = v @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
        v = v.detach().cpu().numpy()
        f = hand_model.mesh[link_name]['visual_faces'].detach().cpu().numpy()
        
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], 
            i=f[:, 0], j=f[:, 1], k=f[:, 2], 
            color='lightgray', opacity=0.3, 
            name=link_name, hovertemplate='%{name}'
        ))

        # ================= 👑 渲染该连杆的局部坐标系 =================
        # 将局部坐标系的4个点按该连杆的FK姿态转换到全局空间
        axes_transformed = hand_model.current_status[link_name].transform_points(local_axes_base)
        if len(axes_transformed.shape) == 3: axes_transformed = axes_transformed[0]
        axes_transformed = axes_transformed @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
        axes_transformed = axes_transformed.detach().cpu().numpy()

        origin = axes_transformed[0]
        pt_x = axes_transformed[1]
        pt_y = axes_transformed[2]
        pt_z = axes_transformed[3]

        # 隐藏 Legend 以免右边栏被塞爆，但保留 hoverinfo 用于鼠标悬停查看
        fig.add_trace(go.Scatter3d(x=[origin[0], pt_x[0]], y=[origin[1], pt_x[1]], z=[origin[2], pt_x[2]], mode='lines', line=dict(color='red', width=4), name=f"{link_name} +X", hoverinfo='name', showlegend=False))
        fig.add_trace(go.Scatter3d(x=[origin[0], pt_y[0]], y=[origin[1], pt_y[1]], z=[origin[2], pt_y[2]], mode='lines', line=dict(color='green', width=4), name=f"{link_name} +Y", hoverinfo='name', showlegend=False))
        fig.add_trace(go.Scatter3d(x=[origin[0], pt_z[0]], y=[origin[1], pt_z[1]], z=[origin[2], pt_z[2]], mode='lines', line=dict(color='blue', width=4), name=f"{link_name} +Z", hoverinfo='name', showlegend=False))
        # =============================================================

    # 渲染接触点 (红点)
    if has_contact:
        pts = hand_model.mesh[link_name]['contact_candidates']
        n_pts = pts.shape[0]
        
        if n_pts > 0:
            total_contact += n_pts
            pts_transformed = hand_model.current_status[link_name].transform_points(pts)
            if len(pts_transformed.shape) == 3: pts_transformed = pts_transformed[0]
            pts_transformed = pts_transformed @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
            pts_transformed = pts_transformed.detach().cpu().numpy()

            fig.add_trace(go.Scatter3d(
                x=pts_transformed[:, 0], y=pts_transformed[:, 1], z=pts_transformed[:, 2],
                mode='markers',
                marker=dict(size=4, color='red', opacity=1.0),
                name=f"{link_name} (接触点)"
            ))
            print(f"✅ {link_name.ljust(25)} : 渲染了 {n_pts} 个接触点")
        else:
            print(f"⚠️ {link_name.ljust(25)} : 提取到 0 个接触点，请检查配置")
    elif has_mesh:
        print(f"➖ {link_name.ljust(25)} : 无主动接触点需求 (正常忽略)")

print("-" * 50)
# 画全局坐标轴防迷路 (使用更粗的线条区分局部坐标系)
fig.add_trace(go.Scatter3d(x=[0, 0.1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=8), name='Global +X'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0.1], z=[0, 0], mode='lines', line=dict(color='green', width=8), name='Global +Y'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 0.1], mode='lines', line=dict(color='blue', width=8), name='Global +Z'))

fig.update_layout(
    title=f"L20 右手接触点与局部坐标轴验收 - 共 {total_contact} 🔴",
    scene=dict(aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=40)
)

output_html = "l20_right_contact_fk_vis.html"
fig.write_html(output_html)
print(f"🎉 网页已保存至: {os.path.abspath(output_html)}")
print("💡 提示: 红色线条为局部 X 轴 (0)，绿色为局部 Y 轴 (1)，蓝色为局部 Z 轴 (2)。")