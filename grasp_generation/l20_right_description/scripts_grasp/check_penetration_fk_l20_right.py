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

print("🔧 [L20 右手] 正在加载 HandModel 与防穿模装甲 FK 链条...")

hand_model = HandModel(
    urdf_path='l20_right_description/linkerhand_l20_right.urdf',
    contact_points_path='l20_right_description/contact_points.json',  # 占位符
    penetration_points_path='l20_right_description/penetration_points.json',
    device=device
)

n_dofs = hand_model.n_dofs
hand_pose = torch.zeros((1, 9 + n_dofs), dtype=torch.float, device=device)
hand_pose[0, 3] = 1.0  # Ortho6D x-axis
hand_pose[0, 7] = 1.0  # Ortho6D y-axis

hand_model.set_parameters(hand_pose)

fig = go.Figure()
total_pen = 0

print("\n🚀 正在进行空间变换与装甲点渲染...")
print("-" * 50)

for link_name in hand_model.mesh:
    # 渲染半透明骨架
    has_mesh = 'visual_vertices' in hand_model.mesh[link_name]
    has_pen = 'penetration_points' in hand_model.mesh[link_name]
    
    if has_mesh:
        v = hand_model.current_status[link_name].transform_points(hand_model.mesh[link_name]['visual_vertices'])
        if len(v.shape) == 3: v = v[0]
        v = v @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
        v = v.detach().cpu().numpy()
        f = hand_model.mesh[link_name]['visual_faces'].detach().cpu().numpy()
        
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], 
            i=f[:, 0], j=f[:, 1], k=f[:, 2], 
            color='lightblue', opacity=0.2, 
            name=link_name, hovertemplate='%{name}'
        ))

    # 渲染穿透装甲点 (蓝点)
    if has_pen:
        pts_pen = hand_model.mesh[link_name]['penetration_points']
        n_pen = pts_pen.shape[0]
        
        if n_pen > 0:
            total_pen += n_pen
            pts_pen_transformed = hand_model.current_status[link_name].transform_points(pts_pen)
            if len(pts_pen_transformed.shape) == 3: pts_pen_transformed = pts_pen_transformed[0]
            pts_pen_transformed = pts_pen_transformed @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
            pts_pen_transformed = pts_pen_transformed.detach().cpu().numpy()

            fig.add_trace(go.Scatter3d(
                x=pts_pen_transformed[:, 0], y=pts_pen_transformed[:, 1], z=pts_pen_transformed[:, 2],
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.6),
                name=f"{link_name} (装甲点)"
            ))
            print(f"✅ {link_name.ljust(25)} : 渲染了 {n_pen} 个装甲点")
        else:
             print(f"⚠️ {link_name.ljust(25)} : 无装甲点 (跳过)")
    elif has_mesh:
         print(f"🚫 {link_name.ljust(25)} : 未配置装甲点 (可能被列入黑名单，如 base1/base2)")

print("-" * 50)
# 画坐标轴防迷路 (红X, 绿Y, 蓝Z)
fig.add_trace(go.Scatter3d(x=[0, 0.1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=6), name='+X'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0.1], z=[0, 0], mode='lines', line=dict(color='green', width=6), name='+Y'))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 0.1], mode='lines', line=dict(color='blue', width=6), name='+Z'))

fig.update_layout(
    title=f"L20 右手防穿模装甲 (Penetration) FK 验收 - 共 {total_pen} 🔵",
    scene=dict(aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=40)
)

output_html = "l20_right_penetration_fk_vis.html"
fig.write_html(output_html)
print(f"🎉 防穿模网页已保存至: {os.path.abspath(output_html)}")