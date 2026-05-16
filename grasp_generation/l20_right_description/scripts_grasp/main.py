"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: Entry of the program
"""

import os

# os.chdir(os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import transforms3d
import math
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.logger import Logger
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="1", type=str)

parser.add_argument('--object_code_list', default=['adjustable_wrench'], type=list) 
parser.add_argument('--name', default='exp_33', type=str)
parser.add_argument('--n_contact', default=5, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--n_iter', default=6000, type=int)
# hyper parameters
parser.add_argument('--switch_possibility', default=0.5, type=float)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--noise_size', default=0.005, type=float)
parser.add_argument('--stepsize_period', default=50, type=int)
parser.add_argument('--starting_temperature', default=18, type=float)
parser.add_argument('--annealing_period', default=30, type=int)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--w_dis', default=100.0, type=float)
parser.add_argument('--w_pen', default=100.0, type=float)
parser.add_argument('--w_spen', default=30.0, type=float)
parser.add_argument('--w_joints', default=1.0, type=float)
# initialization settings
parser.add_argument('--jitter_strength', default=0.05, type=float)     # 小碎步防穿模
parser.add_argument('--distance_lower', default=0.05, type=float)      # 贴脸出生
parser.add_argument('--distance_upper', default=0.1, type=float)       # 缩小搜索范围
parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
parser.add_argument('--theta_upper', default=math.pi / 2, type=float)
# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# prepare models

total_batch_size = len(args.object_code_list) * args.batch_size

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)

# =========================================================================
# 👑 修改点 1：全面加载 L20 灵巧手的 URDF 和两套 JSON 点阵文件
# =========================================================================
hand_model = HandModel(
    urdf_path='l20_right_description/linkerhand_l20_right.urdf',
    contact_points_path='l20_right_description/contact_points.json', 
    penetration_points_path='l20_right_description/penetration_points.json', 
    n_surface_points=1000, 
    device=device
)

object_model = ObjectModel(
    data_root_path='../data/meshdata',
    batch_size_each=args.batch_size,
    num_samples=2000, 
    device=device
)
object_model.initialize(args.object_code_list)

# ==========================================
# 强行覆写自定义物体的物理尺寸 (0.2米)
# ==========================================
object_model.object_scale_tensor = torch.ones(
    len(args.object_code_list), 
    args.batch_size, 
    dtype=torch.float, 
    device=device
) * 1.0  # set scale =1.0

# initialize_convex_hull(hand_model, object_model, args)

hand_pose_init, contact_point_indices = initialize_convex_hull(hand_model, object_model, args)
# =========================================================================
# 👑 坐标系原点补偿与旋转校准 (动态自适应中心版)
# =========================================================================
# batch_size = hand_model.hand_pose.shape[0]
# new_pose = hand_model.hand_pose.detach().clone()

# ✅ 改成这两行：直接从刚刚接住的 hand_pose_init 里取数据！
batch_size = hand_pose_init.shape[0]
new_pose = hand_pose_init.detach().clone()

with torch.no_grad():
    original_rot_mat = robust_compute_rotation_matrix_from_ortho6d(new_pose[:, 3:9])
    
    # ---------------------------------------------------------
    # 👑 修改点 2：动态计算手掌心 (适配 L20 的 base_link 名称)
    # 从 hand_model 中直接提取手掌 (base_link) 的所有接触点
    palm_contacts = hand_model.mesh['base_link']['contact_candidates']
    
    if palm_contacts.shape[0] > 0:
        # 如果有吸盘，掌心就是所有吸盘的平均坐标！
        dynamic_center = palm_contacts.mean(dim=0)
    else:
        # 如果当前模型连手掌吸盘都没有，默认回退到原点 (0,0,0)
        dynamic_center = torch.zeros(3, dtype=torch.float, device=device)
        
    # 转换形状以适配矩阵运算 (1, 3, 1) -> (batch_size, 3, 1)
    joint_offset = dynamic_center.unsqueeze(0).unsqueeze(2).expand(batch_size, 3, 1)
    # ---------------------------------------------------------

    # 绕X轴旋转 -90 度 (根据你前面测试的实际朝向保留)
    correction_mat = torch.tensor(
        transforms3d.euler.euler2mat(-math.pi / 2, 0, 0, axes='sxyz'),
        dtype=torch.float, device=device
    ).unsqueeze(0).expand(batch_size, 3, 3)

    # 计算叠加后的全新旋转姿态并赋值
    new_rot_mat = torch.matmul(original_rot_mat, correction_mat)
    new_pose[:, 3:9] = new_rot_mat[:, :, :2].transpose(1, 2).reshape(-1, 6)

    # 用【新】的旋转矩阵和【动态计算】的偏移量进行逆向平移
    global_offset = -torch.bmm(new_rot_mat, joint_offset).squeeze(2)
    new_pose[:, 0:3] += global_offset

# 插回数据线，激活物理引擎
new_pose.requires_grad_(True)
# hand_model.set_parameters(new_pose)
hand_model.set_parameters(new_pose, contact_point_indices)

# =========================================================================
print('n_contact_candidates', hand_model.n_contact_candidates)
print('total batch size', total_batch_size)
hand_pose_st = hand_model.hand_pose.detach()

optim_config = {
    'switch_possibility': args.switch_possibility,
    'starting_temperature': args.starting_temperature,
    'temperature_decay': args.temperature_decay,
    'annealing_period': args.annealing_period,
    'noise_size': args.noise_size,
    'stepsize_period': args.stepsize_period,
    'mu': args.mu,
    'device': device
}
optimizer = Annealing(hand_model, **optim_config)

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'logs'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'logs'), exist_ok=True)
logger_config = {
    'thres_fc': args.thres_fc,
    'thres_dis': args.thres_dis,
    'thres_pen': args.thres_pen
}
logger = Logger(log_dir=os.path.join('../data/experiments', args.name, 'logs'), **logger_config)


# optimize

weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_spen=args.w_spen,
    w_joints=args.w_joints,
)
energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

energy.sum().backward(retain_graph=True)
logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, 0, show=False)

for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
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

        logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, show=False)


# save results
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

joint_names = hand_model.chain.get_joint_parameter_names()

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'results'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'results'), exist_ok=True)
result_path = os.path.join('../data/experiments', args.name, 'results')
os.makedirs(result_path, exist_ok=True)
for i in range(len(args.object_code_list)):
    data_list = []
    for j in range(args.batch_size):
        idx = i * args.batch_size + j
        scale = object_model.object_scale_tensor[i][j].item()
        hand_pose = hand_model.hand_pose[idx].detach().cpu()
        qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos.update(dict(zip(rot_names, euler)))
        qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
        hand_pose = hand_pose_st[idx].detach().cpu()
        qpos_st = dict(zip(joint_names, hand_pose[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos_st.update(dict(zip(rot_names, euler)))
        qpos_st.update(dict(zip(translation_names, hand_pose[:3].tolist())))
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
    np.save(os.path.join(result_path, args.object_code_list[i] + '.npy'), data_list, allow_pickle=True)