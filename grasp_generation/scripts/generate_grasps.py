"""
Last modified date: 2026.05.10
Description: generate grasps in large-scale for LinkerHand L20 (100% Synced with main.py)
"""

import os
import sys

# 稳定切换到 grasp_generation 根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(ROOT_DIR) != 'grasp_generation':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

import argparse
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import math
import random
import transforms3d

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')


def generate(args_list):
    args, object_code_list, id, gpu_list = args_list

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare models
    n_objects = len(object_code_list)

    worker = multiprocessing.current_process()._identity[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[(worker - 1) % len(gpu_list)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 💡 100% 对齐 main.py：加载路径与 SDF 强制开启
    hand_model = HandModel(
        model_path=args.hand_model_path,
        mesh_path=args.mesh_path,
        contact_points_path=args.contact_points_path,
        penetration_points_path=args.penetration_points_path,
        device=device,
        force_mesh_sdf=True
    )

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size_each,
        num_samples=2000, 
        device=device
    )
    object_model.initialize(object_code_list)

    initialize_convex_hull(hand_model, object_model, args)
    
    hand_pose_st = hand_model.hand_pose.detach()

    optim_config = {
        'switch_possibility': args.switch_possibility,
        'starting_temperature': args.starting_temperature,
        'temperature_decay': args.temperature_decay,
        'annealing_period': args.annealing_period,
        'step_size': args.step_size,
        'stepsize_period': args.stepsize_period,
        'mu': args.mu,
        'device': device
    }
    optimizer = Annealing(hand_model, **optim_config)

    # optimize
    weight_dict = dict(
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_spen=args.w_spen,
        w_joints=args.w_joints,
    )
    
    # 💡 100% 对齐 main.py：设置 verbose=True 获取完整的 6 个返回值，防止解包报错
    energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    energy.sum().backward(retain_graph=True)

    for step in range(1, args.n_iter + 1):
        # if step % 1000 == 0:
            # print(f"[GPU {os.environ['CUDA_VISIBLE_DEVICES']}] {object_code_list[0]} 正在优化... ({step}/{args.n_iter} 步)")
        print(f"[GPU {os.environ['CUDA_VISIBLE_DEVICES']}] {object_code_list[0]} 正在优化... ({step}/{args.n_iter} 步)")
        s = optimizer.try_step()

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]

    # =========================================================================
    # save results (100% 对齐 main.py 的纯净 L20 字典格式)
    # =========================================================================
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    
    # 💡 动态获取关节名
    joint_names = hand_model.joints_names

    for i, object_code in enumerate(object_code_list):
        data_list = []
        for j in range(args.batch_size_each):
            idx = i * args.batch_size_each + j
            scale = object_model.object_scale_tensor[i][j].item()
            
            # ---------- 获取 L20 真实的执行位姿 ----------
            hand_pose = hand_model.hand_pose[idx].detach().cpu()
            qpos = dict(zip(joint_names, hand_pose[9:9+len(joint_names)].tolist()))
            
            # 🔥 修复 1: 强制转换为 numpy 数组
            rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0].numpy()
            euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
            qpos.update(dict(zip(rot_names, euler)))
            qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))

            # ---------- 初始姿态处理 ----------
            # 🔥 修复 2: 隔离变量名 hand_pose_st_idx
            hand_pose_st_idx = hand_pose_st[idx].detach().cpu()
            qpos_st = dict(zip(joint_names, hand_pose_st_idx[9:9+len(joint_names)].tolist()))
            
            # 🔥 修复 3: 强制转换为 numpy 数组
            rot_st = robust_compute_rotation_matrix_from_ortho6d(hand_pose_st_idx[3:9].unsqueeze(0))[0].numpy()
            euler_st = transforms3d.euler.mat2euler(rot_st, axes='sxyz')
            qpos_st.update(dict(zip(rot_names, euler_st)))
            qpos_st.update(dict(zip(translation_names, hand_pose_st_idx[:3].tolist())))

            # 💡 精简输出，直接对齐 main.py
            data_list.append(dict(
                scale=scale,
                qpos=qpos,
                qpos_st=qpos_st,
                energy=energy[idx].item(),
                E_fc=E_fc[idx].item(),
                E_dis=E_dis[idx].item(),
                E_pen=E_pen[idx].item(),
                E_spen=E_spen[idx].item(),
                E_joints=E_joints[idx].item(),
            ))
        np.save(os.path.join(args.result_path, object_code + '.npy'), data_list, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--result_path', default="../data/l20_graspdata", type=str)
    parser.add_argument('--data_root_path', default="../data/meshdata", type=str)
    parser.add_argument('--object_code_list', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--todo', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    
    # 💡 100% 对齐 main.py
    parser.add_argument('--n_contact', default=4, type=int)
    parser.add_argument('--batch_size_each', default=128, type=int)
    parser.add_argument('--max_total_batch_size', default=1000, type=int)
    parser.add_argument('--n_iter', default=6000, type=int)
    
    # 💡 L20 专属路径对齐 main.py
    parser.add_argument('--hand_model_path', default='mjcf/linkerhand_l20_right.urdf', type=str)
    parser.add_argument('--mesh_path', default='mjcf/meshes', type=str)
    parser.add_argument('--contact_points_path', default='mjcf/contact_points.json', type=str)
    parser.add_argument('--penetration_points_path', default='mjcf/penetration_points.json', type=str)

    # hyper parameters
    parser.add_argument('--switch_possibility', default=0.5, type=float)
    parser.add_argument('--mu', default=0.98, type=float)
    parser.add_argument('--step_size', default=0.005, type=float)
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=18, type=float)
    parser.add_argument('--annealing_period', default=30, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    parser.add_argument('--w_dis', default=100.0, type=float)
    parser.add_argument('--w_pen', default=100.0, type=float)
    parser.add_argument('--w_spen', default=10.0, type=float)
    parser.add_argument('--w_joints', default=1.0, type=float)
    
    # 💡 初始化参数 100% 对齐 main.py
    parser.add_argument('--jitter_strength', default=0.1, type=float)
    parser.add_argument('--distance_lower', default=0.05, type=float)
    parser.add_argument('--distance_upper', default=0.10, type=float)
    parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
    parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
    
    # energy thresholds
    parser.add_argument('--thres_fc', default=0.3, type=float)
    parser.add_argument('--thres_dis', default=0.005, type=float)
    parser.add_argument('--thres_pen', default=0.001, type=float)

    args = parser.parse_args()

    # 安全获取 GPU 列表
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_list = cuda_visible.split(",")
    print(f'GPU 列表: {gpu_list}')

    # check whether arguments are valid and process arguments
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    if not os.path.exists(args.data_root_path):
        raise ValueError(f'data_root_path {args.data_root_path} doesn\'t exist')
    
    if (args.object_code_list is not None) + args.all != 1:
        raise ValueError('exactly one among \'object_code_list\' \'all\' should be specified')
    
    if args.todo:
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line[:-1] for line in lines]
    else:
        # 获取合法目录
        object_code_list_all = [d for d in os.listdir(args.data_root_path) if os.path.isdir(os.path.join(args.data_root_path, d))]
    
    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if not set(object_code_list).issubset(set(object_code_list_all)):
            raise ValueError('object_code_list isn\'t a subset of dirs in data_root_path')
    else:
        object_code_list = object_code_list_all
    
    if not args.overwrite:
        for object_code in object_code_list.copy():
            if os.path.exists(os.path.join(args.result_path, object_code + '.npy')):
                object_code_list.remove(object_code)

    if args.batch_size_each > args.max_total_batch_size:
        raise ValueError(f'batch_size_each {args.batch_size_each} should be smaller than max_total_batch_size {args.max_total_batch_size}')
    
    print(f'待生成抓取的物体数量: {len(object_code_list)}')
    
    if len(object_code_list) == 0:
        print("所有物体的抓取数据均已存在，退出生成。 (如需重写，请加上 --overwrite)")
        sys.exit(0)

    # generate
    random.seed(args.seed)
    random.shuffle(object_code_list)
    
    # 🔥 修复 1: 强制每个任务只分配 1 个物体，防止显存爆炸，且让进度条动起来！
    objects_each = 1
    object_code_groups = [[obj] for obj in object_code_list]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append((args, object_code_group, id + 1, gpu_list))

    # 使用多进程映射任务
    with multiprocessing.Pool(len(gpu_list)) as p:
        it = tqdm(p.imap_unordered(generate, process_args), total=len(process_args), desc='批量生成中...', maxinterval=1000)
        list(it)

    print(f"\n🎉 批量抓取生成完成！结果保存在: {os.path.abspath(args.result_path)}")