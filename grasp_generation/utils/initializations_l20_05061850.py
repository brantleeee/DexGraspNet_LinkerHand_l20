"""
Last modified date: 2026.05.02
Description: initializations (Python-based Math Fix for L20 / Cross-Embodiment)
"""

import torch
import transforms3d
import math
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
import numpy as np


def initialize_convex_hull(hand_model, object_model, args):
    """
    Initialize grasp translation, rotation, joint angles, and contact point indices
    """
    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each

    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    for i in range(n_objects):
        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces()]
        
        # 安全补丁：防止除0
        norm = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices += 0.2 * vertices / np.maximum(norm, 1e-8)
        
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        # PyTorch3D 要求 faces 为 long 类型
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * batch_size_each)
        p = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=batch_size_each)[0][0]
        closest_points, _, _ = mesh_origin.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        
        diff = closest_points - p
        n = diff / diff.norm(dim=1).clamp_min(1e-8).unsqueeze(1)

        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        deviate_theta = args.theta_lower + (args.theta_upper - args.theta_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        process_theta = 2 * math.pi * torch.rand([batch_size_each], dtype=torch.float, device=device)
        rotate_theta = 2 * math.pi * torch.rand([batch_size_each], dtype=torch.float, device=device)

        rotation_local = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        rotation_global = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_size_each):
            # 🔥 修复：加上 .item()，确保传给 transforms3d 的是纯粹的浮点数
            rotation_local[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    process_theta[j].item(), 
                    deviate_theta[j].item(), 
                    rotate_theta[j].item(), 
                    axes='rzxz'
                ), 
                dtype=torch.float, device=device
            )
            # 安全补丁：防止 acos 越界
            val = torch.clamp(n[j, 2], -1.0, 1.0).item()
            # 🔥 修复：math.atan2 同样要求输入标准 float，必须加 .item()
            rotation_global[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    math.atan2(n[j, 1].item(), n[j, 0].item()) - math.pi / 2, 
                    -math.acos(val), 
                    0, 
                    axes='rzxz'
                ), 
                dtype=torch.float, device=device
            )
            
        # translation[i * batch_size_each: (i + 1) * batch_size_each] = p - distance.unsqueeze(1) * (rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)).squeeze(2)
        # （这是你原本的那行 translation 计算代码，把它注释掉或删掉）
        # translation[i * batch_size_each: ...] = p - distance.unsqueeze(1) ...

        # =========================================================================
        # 💡 核心坐标系与手腕杠杆偏移补偿补丁
        # =========================================================================
        # 1. 修正绕自身Z轴旋转的 -90 度
        rotation_shadow = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device)
        linker_z_fix_deg = -90.0  # 强制写死，确保 L20 旋转对齐
        linker_fix = torch.tensor(transforms3d.euler.euler2mat(0, 0, math.radians(linker_z_fix_deg), axes='sxyz'), dtype=torch.float, device=device)
        
        # 2. 计算最终的旋转矩阵
        rotation_hand = rotation_shadow @ linker_fix
        R_final = rotation_global @ rotation_local @ rotation_hand
        rotation[i * batch_size_each: (i + 1) * batch_size_each] = R_final
        
        # 3. 计算“理想的掌心”应该在宇宙中的哪个绝对坐标位置
        pos_palm = p - distance.unsqueeze(1) * (rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)).squeeze(2)
        
        # 4. 🔥 抵消 L20 的手腕物理偏移
        # 假设 L20 的掌心相对于手腕 (base_link) 在 Z 轴正方向 15 厘米处。
        # 这里你可以根据你的 URDF 实际长度微调这个 [0.0, 0.0, 0.15] 的数值
        palm_offset = torch.tensor([0.0, 0.0, 0.03], dtype=torch.float, device=device).reshape(1, 3, 1)
        
        # 真正的 base_link 坐标 = 掌心坐标 - (旋转后的偏移向量)
        translation[i * batch_size_each: (i + 1) * batch_size_each] = pos_palm - (R_final @ palm_offset).squeeze(2)
        # =========================================================================
        # =========================================================================
        # 💡 核心坐标系偏移补丁 (Python 层解决)
        # =========================================================================
        rotation_shadow = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device)
        
        linker_z_fix_deg = getattr(args, 'linker_z_fix_deg', -90.0)
        linker_fix = torch.tensor(transforms3d.euler.euler2mat(0, 0, math.radians(linker_z_fix_deg), axes='sxyz'), dtype=torch.float, device=device)
        
        # 真实给到物理引擎的姿态：先按 ShadowHand 的方式对准物体，然后再绕自己的Z轴转 -90 度
        rotation_hand = rotation_shadow @ linker_fix
        # =========================================================================
        
        rotation[i * batch_size_each: (i + 1) * batch_size_each] = rotation_global @ rotation_local @ rotation_hand
    
    # 💡 动态获取关节信息，避免写死 22 维导致 L20 报错
    joint_angles_mu = 0.5 * (hand_model.joints_lower + hand_model.joints_upper)
    joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles_sigma = torch.clamp(joint_angles_sigma, min=1e-6)
    
    joint_angles = torch.zeros([total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(joint_angles[:, i], joint_angles_mu[i].item(), joint_angles_sigma[i].item(), (hand_model.joints_lower[i] - 1e-6).item(), (hand_model.joints_upper[i] + 1e-6).item())

    hand_pose = torch.cat([
        translation,
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose.requires_grad_()

    contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, args.n_contact], device=device)
    hand_model.set_parameters(hand_pose, contact_point_indices)