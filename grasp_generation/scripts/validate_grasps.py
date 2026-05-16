"""
Modified for LinkerHand L20
Description: validate grasps on Isaac simulator for L20
"""

import os
import sys

# 稳定切换到 grasp_generation 根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(ROOT_DIR) != 'grasp_generation':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from utils.isaac_validator import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
from utils.hand_model import HandModel
from utils.object_model import ObjectModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int) # 默认改回 0
    parser.add_argument('--val_batch', default=500, type=int)
    parser.add_argument('--mesh_path', default="../data/meshdata", type=str)
    
    # 💡 路径默认指向 L20 数据
    parser.add_argument('--grasp_path', default="../data/l20_graspdata", type=str)
    parser.add_argument('--result_path', default="../data/l20_dataset", type=str) 
    
    parser.add_argument('--object_code', default="core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03", type=str)
    parser.add_argument('--index', type=int)
    parser.add_argument('--no_force', action='store_true')
    parser.add_argument('--thres_cont', default=0.001, type=float)
    parser.add_argument('--dis_move', default=0.001, type=float)
    parser.add_argument('--grad_move', default=500, type=float)
    parser.add_argument('--penetration_threshold', default=0.001, type=float)

    # 💡 L20 专属物理模型路径
    parser.add_argument('--hand_model_path', default='mjcf/linkerhand_l20_right.urdf', type=str)
    parser.add_argument('--mesh_path_hand', default='mjcf/meshes', type=str)
    parser.add_argument('--contact_points_path', default='mjcf/contact_points.json', type=str)
    parser.add_argument('--penetration_points_path', default='mjcf/penetration_points.json', type=str)

    args = parser.parse_args()

    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("CUDA_VISIBLE_DEVICES")
        
    os.makedirs(args.result_path, exist_ok=True)

    if not args.no_force:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # =========================================================================
        # 1. 优先加载 HandModel 以动态获取 L20 属性
        # =========================================================================
        hand_model = HandModel(
            model_path=args.hand_model_path,
            mesh_path=args.mesh_path_hand,
            contact_points_path=args.contact_points_path,
            penetration_points_path=args.penetration_points_path,
            n_surface_points=2000,
            device=device,
            force_mesh_sdf=True
        )
        # 动态获取真实关节名和连杆总数，彻底消灭硬编码
        joint_names = hand_model.joints_names
        num_links = len(hand_model.mesh)
        
        data_dict = np.load(os.path.join(args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
        batch_size = data_dict.shape[0]
        hand_state = []
        scale_tensor = []
        
        for i in range(batch_size):
            qpos = data_dict[i]['qpos']
            scale = data_dict[i]['scale']
            rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
            rot = rot[:, :2].T.ravel().tolist()
            # 使用 get 安全读取
            hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [
                qpos.get(name, 0.0) for name in joint_names], dtype=torch.float, device=device)
            hand_state.append(hand_pose)
            scale_tensor.append(scale)
            
        hand_state = torch.stack(hand_state).to(device).requires_grad_()
        scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)

        hand_model.set_parameters(hand_state)
        
        # object model
        object_model = ObjectModel(
            data_root_path=args.mesh_path,
            batch_size_each=batch_size,
            num_samples=0,
            device=device
        )
        object_model.initialize([args.object_code])
        object_model.object_scale_tensor = scale_tensor

        # =========================================================================
        # 2. 动态计算接触点 (取代原版硬编码的 19 个连杆)
        # =========================================================================
        contact_points_hand = torch.zeros((batch_size, num_links, 3)).to(device)
        contact_normals = torch.zeros((batch_size, num_links, 3)).to(device)

        for i, link_name in enumerate(hand_model.mesh):
            if len(hand_model.mesh[link_name]['surface_points']) == 0:
                continue
            surface_points = hand_model.current_status[link_name].transform_points(
                hand_model.mesh[link_name]['surface_points']).expand(batch_size, -1, 3)
            surface_points = surface_points @ hand_model.global_rotation.transpose(
                1, 2) + hand_model.global_translation.unsqueeze(1)
            distances, normals = object_model.cal_distance(surface_points)
            
            nearest_point_index = distances.argmax(dim=1)
            nearest_distances = torch.gather(distances, 1, nearest_point_index.unsqueeze(1))
            nearest_points_hand = torch.gather(surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            nearest_normals = torch.gather(normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            
            admited = -nearest_distances < args.thres_cont
            admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
            
            contact_points_hand[:, i:i+1, :] = torch.where(admited, nearest_points_hand, contact_points_hand[:, i:i+1, :])
            contact_normals[:, i:i+1, :] = torch.where(admited, nearest_normals, contact_normals[:, i:i+1, :])

        target_points = contact_points_hand + contact_normals * args.dis_move
        loss = (target_points.detach().clone() - contact_points_hand).square().sum()
        loss.backward()
        with torch.no_grad():
            hand_state[:, 9:] += hand_state.grad[:, 9:] * args.grad_move
            hand_state.grad.zero_()

    # =========================================================================
    # 3. Isaac Gym 物理仿真模块配置
    # =========================================================================
    sim = IsaacValidator(gpu=args.gpu)
    if (args.index is not None):
        sim = IsaacValidator(gpu=args.gpu, mode="gui")

    data_dict = np.load(os.path.join(args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
    batch_size = data_dict.shape[0]
    scale_array = []
    hand_poses = []
    rotations = []
    translations = []
    E_pen_array = []
    
    # 获取 L20 真实的 joint names (如果没有走 no_force 逻辑)
    if args.no_force:
        hand_model_temp = HandModel(model_path=args.hand_model_path, mesh_path=args.mesh_path_hand, contact_points_path=args.contact_points_path, penetration_points_path=args.penetration_points_path, device='cpu', force_mesh_sdf=True)
        joint_names = hand_model_temp.joints_names
        
    for i in range(batch_size):
        qpos = data_dict[i]['qpos']
        scale = data_dict[i]['scale']
        rot = [qpos[name] for name in rot_names]
        rot = transforms3d.euler.euler2quat(*rot)
        rotations.append(rot)
        translations.append(np.array([qpos[name] for name in translation_names]))
        hand_poses.append(np.array([qpos.get(name, 0.0) for name in joint_names]))
        scale_array.append(scale)
        E_pen_array.append(data_dict[i]["E_pen"])
        
    E_pen_array = np.array(E_pen_array)
    if not args.no_force:
        # 使用逆运动学微调过的关节点位
        hand_poses = hand_state[:, 9:].detach().cpu().numpy()

    # 💡 核心修复：分离物理资产所在的目录与文件名，让 Isaac Validator 精准加载 L20
    hand_asset_root = os.path.dirname(args.hand_model_path)
    hand_asset_file = os.path.basename(args.hand_model_path)

    if (args.index is not None):
        # 加载 L20 物理资产
        sim.set_asset(hand_asset_root, hand_asset_file, os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
        index = args.index
        sim.add_env_single(rotations[index], translations[index], hand_poses[index], scale_array[index], 0)
        result = sim.run_sim()
        print(result)
    else:
        simulated = np.zeros(batch_size, dtype=np.bool8)
        offset = 0
        result = []
        for batch in range(batch_size // args.val_batch):
            offset_ = min(offset + args.val_batch, batch_size)
            
            # 加载 L20 物理资产
            sim.set_asset(hand_asset_root, hand_asset_file, os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
            for index in range(offset, offset_):
                sim.add_env(rotations[index], translations[index], hand_poses[index], scale_array[index])
            result = [*result, *sim.run_sim()]
            sim.reset_simulator()
            offset = offset_
            
        for i in range(batch_size):
            simulated[i] = np.array(sum(result[i * 6:(i + 1) * 6]) == 6)

        estimated = E_pen_array < args.penetration_threshold
        valid = simulated * estimated
        print(
            f'estimated: {estimated.sum().item()}/{batch_size}, '
            f'simulated: {simulated.sum().item()}/{batch_size}, '
            f'valid: {valid.sum().item()}/{batch_size}')
            
        result_list = []
        for i in range(batch_size):
            if (valid[i]):
                new_data_dict = {}
                new_data_dict["qpos"] = data_dict[i]["qpos"]
                new_data_dict["scale"] = data_dict[i]["scale"]
                result_list.append(new_data_dict)
                
        np.save(os.path.join(args.result_path, args.object_code + '.npy'), result_list, allow_pickle=True)
        print(f"✅ 验证完成并保存可用抓取至: {args.result_path}/{args.object_code}.npy")
        
    sim.destroy()