import os
import sys
import torch
import numpy as np
import plotly.graph_objects as go

# 兼容各种路径运行，确保能找到 utils
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hand_model import HandModel

# 强制使用 CPU 进行纯数据推演
device = 'cpu'

print("🔧 正在加载 Allegro Hand 的 URDF 与正运动学 (FK) 链条...")

# =========================================================================
# 1. 初始化 HandModel (指向 Allegro 目录)
# 根据你的 tree 输出，我们这里使用右手模型 (right.urdf)
# =========================================================================
hand_model = HandModel(
    urdf_path='allegro_hand_description/allegro_hand_description_right.urdf',
    contact_points_path='allegro_hand_description/contact_points.json',
    device=device
)

# =========================================================================
# 2. 构造严格的“零位姿态” (Zero Pose)
# 根部旋转使用单位正交矩阵 (不作任何空间旋转)，各关节角度设为 0
# =========================================================================
n_dofs = hand_model.n_dofs
hand_pose = torch.zeros((1, 9 + n_dofs), dtype=torch.float, device=device)
hand_pose[0, 3] = 1.0  # Ortho6D 的 X 轴方向 [1, 0, 0]
hand_pose[0, 7] = 1.0  # Ortho6D 的 Y 轴方向 [0, 1, 0]

# 驱动引擎计算各连杆在空间的绝对变换矩阵
hand_model.set_parameters(hand_pose)

fig = go.Figure()
total_points = 0

print("\n🚀 正在进行 Allegro 空间变换与渲染...")

for link_name in hand_model.mesh:
    # -----------------------------------------------------
    # 渲染手部的半透明 3D 网格 (Mesh)
    # -----------------------------------------------------
    if 'visual_vertices' in hand_model.mesh[link_name]:
        v = hand_model.current_status[link_name].transform_points(hand_model.mesh[link_name]['visual_vertices'])
        if len(v.shape) == 3: v = v[0]
        v = v @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
        v = v.detach().cpu().numpy()
        f = hand_model.mesh[link_name]['visual_faces'].detach().cpu().numpy()
        
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], 
            i=f[:, 0], j=f[:, 1], k=f[:, 2], 
            color='lightgray', opacity=0.3,  # 灰色半透明，模拟金属质感
            name=link_name, hovertemplate='%{name}'
        ))

    # -----------------------------------------------------
    # 渲染经过 FK 变换的引力接触点 (Contact Points)
    # -----------------------------------------------------
    if 'contact_candidates' in hand_model.mesh[link_name]:
        pts = hand_model.mesh[link_name]['contact_candidates']
        n_pts = pts.shape[0]
        
        if n_pts > 0:
            total_points += n_pts
            # 👑 核心：应用正运动学矩阵，将局部点转移到世界坐标系
            pts_transformed = hand_model.current_status[link_name].transform_points(pts)
            if len(pts_transformed.shape) == 3: pts_transformed = pts_transformed[0]
            pts_transformed = pts_transformed @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
            pts_transformed = pts_transformed.detach().cpu().numpy()

            fig.add_trace(go.Scatter3d(
                x=pts_transformed[:, 0], 
                y=pts_transformed[:, 1], 
                z=pts_transformed[:, 2],
                mode='markers',
                marker=dict(size=4, color='red', opacity=0.9),
                name=f"{link_name} (吸盘)"
            ))

# =========================================================================
# 3. 👑 核心对齐基准：绘制世界坐标系原点与三根坐标轴
# =========================================================================
# 红线: +X, 绿线: +Y, 蓝线: +Z
axis_len = 0.15
fig.add_trace(go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines+text', line=dict(color='red', width=8), text=['', '+X (Red)'], textposition='top center', name='+X Axis'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, axis_len], z=[0, 0], mode='lines+text', line=dict(color='green', width=8), text=['', '+Y (Green)'], textposition='top center', name='+Y Axis'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len], mode='lines+text', line=dict(color='blue', width=8), text=['', '+Z (Blue)'], textposition='top center', name='+Z Axis'))

fig.update_layout(
    title=f"Allegro Hand 标准零位坐标系探针 (包含 FK 与 {total_points} 个吸盘点)",
    scene=dict(
        aspectmode='data',
        xaxis_title='X (Red)',
        yaxis_title='Y (Green)',
        zaxis_title='Z (Blue)'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

output_html = "allegro_fk_vis.html"
fig.write_html(output_html)
print(f"\n🎉 Allegro 渲染完成！请下载并在浏览器打开: {os.path.abspath(output_html)}")