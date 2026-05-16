"""
Last modified date: 2026.05.02
Description: LinkerHand L20 initialization for DexGraspNet
"""

import torch
import transforms3d
import math
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
import numpy as np
import torch.nn.functional


def get_l20_rotation_hand(args, device):
    """
    既然 URDF 已经对齐了 ShadowHand 语义，这里直接返回标准的 canonical grasp rotation 即可。
    """
    rotation_hand_shadow = torch.tensor(
        transforms3d.euler.euler2mat(
            0,
            -np.pi / 3,
            0,
            axes='rzxz'
        ),
        dtype=torch.float,
        device=device
    )
    
    # 删掉了 linker_z_fix 相关的计算
    return rotation_hand_shadow


def get_l20_initial_joint_mu(hand_model, args):
    """
    LinkerHand L20 不使用 ShadowHand 的 22 维 joint_angles_mu。

    策略：
    1. 默认以 0 为初值；
    2. 如果 0 不在关节范围内，则使用关节范围中点；
    3. 用 l20_close_ratio 给一点轻微闭合趋势；
    4. clamp 到关节范围内。

    这样不假设具体关节名，适合 L20 的 21 DOF。
    """

    lower = hand_model.joints_lower
    upper = hand_model.joints_upper

    mu = torch.zeros_like(lower)

    zero_invalid = (mu < lower) | (mu > upper)
    mu[zero_invalid] = 0.5 * (lower[zero_invalid] + upper[zero_invalid])

    close_ratio = getattr(args, "l20_close_ratio", 0.25)
    mu = mu + close_ratio * (upper - mu)

    mu = torch.clamp(mu, lower + 1e-6, upper - 1e-6)

    return mu


def initialize_convex_hull(hand_model, object_model, args):
    """
    Initialize LinkerHand L20 grasp translation, rotation, joint angles,
    and contact point indices.

    Parameters
    ----------
    hand_model:
        LinkerHand L20 HandModel from linkerhand_l20_right.urdf.
    object_model:
        DexGraspNet ObjectModel.
    args:
        Namespace from main.py.
    """

    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each

    if hand_model.n_dofs <= 0:
        raise RuntimeError("LinkerHand L20 has no movable DOFs.")

    if hand_model.n_contact_candidates <= 0:
        raise RuntimeError("LinkerHand L20 has no contact candidates.")

    translation = torch.zeros(
        [total_batch_size, 3],
        dtype=torch.float,
        device=device
    )

    rotation = torch.zeros(
        [total_batch_size, 3, 3],
        dtype=torch.float,
        device=device
    )

    rotation_hand = get_l20_rotation_hand(args, device)

    grasp_direction = torch.tensor(
        [0, 0, 1],
        dtype=torch.float,
        device=device
    ).reshape(1, -1, 1)

    for i in range(n_objects):
        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces

        vertices *= object_model.object_scale_tensor[i].max().item()

        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[
            mesh_origin.remove_degenerate_faces()
        ]

        vertices_norm = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices_norm = np.maximum(vertices_norm, 1e-8)

        vertices += 0.2 * vertices / vertices_norm

        mesh = tm.Trimesh(
            vertices=vertices,
            faces=faces
        ).convex_hull

        vertices_tensor = torch.tensor(
            mesh.vertices,
            dtype=torch.float,
            device=device
        )

        faces_tensor = torch.tensor(
            mesh.faces,
            dtype=torch.float,
            device=device
        )

        mesh_pytorch3d = pytorch3d.structures.Meshes(
            vertices_tensor.unsqueeze(0),
            faces_tensor.unsqueeze(0)
        )

        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
            mesh_pytorch3d,
            num_samples=100 * batch_size_each
        )

        p = pytorch3d.ops.sample_farthest_points(
            dense_point_cloud,
            K=batch_size_each
        )[0][0]

        closest_points, _, _ = mesh_origin.nearest.on_surface(
            p.detach().cpu().numpy()
        )

        closest_points = torch.tensor(
            closest_points,
            dtype=torch.float,
            device=device
        )

        n = (closest_points - p)
        n = n / n.norm(dim=1).clamp_min(1e-8).unsqueeze(1)

        distance = args.distance_lower + (
            args.distance_upper - args.distance_lower
        ) * torch.rand(
            [batch_size_each],
            dtype=torch.float,
            device=device
        )

        deviate_theta = args.theta_lower + (
            args.theta_upper - args.theta_lower
        ) * torch.rand(
            [batch_size_each],
            dtype=torch.float,
            device=device
        )

        process_theta = 2 * math.pi * torch.rand(
            [batch_size_each],
            dtype=torch.float,
            device=device
        )

        rotate_theta = 2 * math.pi * torch.rand(
            [batch_size_each],
            dtype=torch.float,
            device=device
        )

        rotation_local = torch.zeros(
            [batch_size_each, 3, 3],
            dtype=torch.float,
            device=device
        )

        rotation_global = torch.zeros(
            [batch_size_each, 3, 3],
            dtype=torch.float,
            device=device
        )

        for j in range(batch_size_each):
            rotation_local[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    process_theta[j],
                    deviate_theta[j],
                    rotate_theta[j],
                    axes='rzxz'
                ),
                dtype=torch.float,
                device=device
            )

            rotation_global[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    math.atan2(n[j, 1], n[j, 0]) - math.pi / 2,
                    -math.acos(torch.clamp(n[j, 2], -1.0, 1.0).item()),
                    0,
                    axes='rzxz'
                ),
                dtype=torch.float,
                device=device
            )

        idx_l = i * batch_size_each
        idx_r = (i + 1) * batch_size_each

        translation[idx_l:idx_r] = p - distance.unsqueeze(1) * (
            rotation_global @ rotation_local @ grasp_direction
        ).squeeze(2)

        rotation[idx_l:idx_r] = (
            rotation_global @ rotation_local @ rotation_hand
        )

    joint_angles_mu = get_l20_initial_joint_mu(hand_model, args)

    joint_angles_sigma = args.jitter_strength * (
        hand_model.joints_upper - hand_model.joints_lower
    )

    joint_angles_sigma = torch.clamp(
        joint_angles_sigma,
        min=1e-6
    )

    joint_angles = torch.zeros(
        [total_batch_size, hand_model.n_dofs],
        dtype=torch.float,
        device=device
    )

    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i],
            mean=joint_angles_mu[i],
            std=joint_angles_sigma[i],
            a=hand_model.joints_lower[i] - 1e-6,
            b=hand_model.joints_upper[i] + 1e-6
        )

    joint_angles = torch.clamp(
        joint_angles,
        hand_model.joints_lower.unsqueeze(0) + 1e-6,
        hand_model.joints_upper.unsqueeze(0) - 1e-6
    )

    hand_pose = torch.cat([
        translation,
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)

    hand_pose.requires_grad_()

    contact_point_indices = torch.randint(
        hand_model.n_contact_candidates,
        size=[total_batch_size, args.n_contact],
        device=device
    )

    hand_model.set_parameters(hand_pose, contact_point_indices)