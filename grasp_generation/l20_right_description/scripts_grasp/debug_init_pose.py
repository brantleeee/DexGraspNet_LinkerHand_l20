import os
import sys
import math
import torch
import transforms3d
import plotly.graph_objects as go

# 兼容路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hand_model import HandModel
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

print("🔬 启动纯净坐标系探测仪 (Zero Pose + Patch)...")

device = 'cpu'
batch_size = 1

# 1. 加载模型 (右手)
hand_model = HandModel(
    urdf_path='l6_right_description/linkerhand_l6_right.urdf',
    contact_points_path='l6_right_description/contact_points.json',
    penetration_points_path='l6_right_description/penetration_points.json',
    device=device
)

# 2. 构造绝对零位姿态 (Zero Pose)
# 把手放在 [0,0,0]，无任何旋转
n_dofs = hand_model.n_dofs
new_pose = torch.zeros((batch_size, 9 + n_dofs), dtype=torch.float, device=device)
new_pose[0, 3] = 1.0  # Ortho6D x轴
new_pose[0, 7] = 1.0  # Ortho6D y轴

# =========================================================================
# 👑 补丁实验室 (Patch Lab) - 数学绝对死锁版
# =========================================================================
original_rot_mat = robust_compute_rotation_matrix_from_ortho6d(new_pose[:, 3:9])

# 1. 提取 URDF 中的真实相对偏移 (钉死中指根部)
joint_offset = torch.tensor([[-0.00066821, 0.0, 0.12068]], dtype=torch.float, device=device).unsqueeze(2)

# --------------------------------------------------
# 2. 旋转修正 (解决 X 轴反了、掌心朝向不对的问题)
# 请每次只保留下面其中【一个】方案取消注释，运行查看红线与掌心方向
# --------------------------------------------------

# 方案 A：绕 Z 轴旋转 180 度 (翻面，X和Y反转，Z不变。最常用于手背翻转)
# rpy = (0, 0, math.pi)
correction_mat = torch.tensor(transforms3d.euler.euler2mat(0, 0, 0, axes='sxyz'), dtype=torch.float, device=device).unsqueeze(0)

# 方案 B：绕 Y 轴旋转 180 度 (翻面，X和Z反转，Y不变)
# rpy = (0, math.pi, 0)
# correction_mat = torch.tensor(transforms3d.euler.euler2mat(0, math.pi, 0, axes='sxyz'), dtype=torch.float, device=device).unsqueeze(0)

# 方案 C：绕 Z 轴转 90 度 (如果是侧面对着，用这个把掌心转正)
# rpy = (0, 0, math.pi / 2)
# correction_mat = torch.tensor(transforms3d.euler.euler2mat(0, 0, math.pi / 2, axes='sxyz'), dtype=torch.float, device=device).unsqueeze(0)

# 方案 D：绕 Z 轴反转 90 度 
# rpy = (0, 0, -math.pi / 2)
# correction_mat = torch.tensor(transforms3d.euler.euler2mat(0, 0, -math.pi / 2, axes='sxyz'), dtype=torch.float, device=device).unsqueeze(0)

# 计算叠加后的全新旋转姿态
new_rot_mat = torch.matmul(original_rot_mat, correction_mat)
new_pose[:, 3:9] = new_rot_mat[:, :, :2].transpose(1, 2).reshape(-1, 6)

# --------------------------------------------------
# 3. 终极绝对原点锁定 (解决原点不对的问题)
# --------------------------------------------------
global_offset = -torch.bmm(new_rot_mat, joint_offset).squeeze(2)
new_pose[:, 0:3] += global_offset
# =========================================================================

# 3. 更新物理状态
hand_model.set_parameters(new_pose)

# 4. 渲染手部
print("🚀 正在生成 3D 渲染...")
hand_plotly = hand_model.get_plotly_data(i=0, opacity=1.0, color='lightblue', visual=True)

# 5. 画出绝对世界坐标轴 (交点即为算法眼里的原点 0,0,0)
axis_len = 0.2
axes_plotly = [
    go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines+text', line=dict(color='red', width=8), text=['', '+X (红)'], textposition='top center', name='+X Axis'),
    go.Scatter3d(x=[0, 0], y=[0, axis_len], z=[0, 0], mode='lines+text', line=dict(color='green', width=8), text=['', '+Y (绿)'], textposition='top center', name='+Y Axis'),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len], mode='lines+text', line=dict(color='blue', width=8), text=['', '+Z (蓝)'], textposition='top center', name='+Z Axis')
]

fig = go.Figure(hand_plotly + axes_plotly)
fig.update_layout(scene_aspectmode='data', title="[调试] 绝对坐标系与补偿测试仪")

output_html = "debug_coordinate_frame.html"
fig.write_html(output_html)
print(f"✅ 坐标系测试网页已导出: {os.path.abspath(output_html)}")