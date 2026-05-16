"""
Last modified date: 2026.05.02
Description: Entry of the program, generate LinkerHand L20 grasps with DexGraspNet
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import transforms3d

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.logger import Logger
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


def get_linker_fix_matrix(linker_z_fix_deg):
    return torch.tensor(
        transforms3d.euler.euler2mat(
            0,
            0,
            math.radians(linker_z_fix_deg),
            axes='sxyz'
        ),
        dtype=torch.float
    )


parser = argparse.ArgumentParser()

# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument(
    '--object_code_list',
    nargs='+',
    default=[
        'core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03'
    ]
)
parser.add_argument('--name', default='l20_exp', type=str)
parser.add_argument('--n_contact', default=4, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--n_iter', default=6000, type=int)

# LinkerHand L20 settings
parser.add_argument(
    '--hand_model_path',
    default='mjcf/linkerhand_l20_right.urdf',
    type=str
)
parser.add_argument('--mesh_path', default='mjcf/meshes', type=str)
parser.add_argument('--contact_points_path', default='mjcf/contact_points.json', type=str)
parser.add_argument('--penetration_points_path', default='mjcf/penetration_points.json', type=str)

# 已验证：LinkerHand L20 需要右乘 Rz(-90°) 才和 ShadowHand 掌心/四指语义一致
# parser.add_argument('--linker_z_fix_deg', default=-90.0, type=float)
parser.add_argument('--linker_z_fix_deg', default=0.0, type=float)
parser.add_argument('--l20_close_ratio', default=0.25, type=float)

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

# initialization settings
parser.add_argument('--jitter_strength', default=0.1, type=float)
parser.add_argument('--distance_lower', default=0.2, type=float)
parser.add_argument('--distance_upper', default=0.3, type=float)
parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
parser.add_argument('--theta_upper', default=math.pi / 6, type=float)

# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

args = parser.parse_args()

# 给 initializations.py 使用
args.hand_type = 'l20'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)

total_batch_size = len(args.object_code_list) * args.batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)

# -------------------------------------------------------------------------
# prepare LinkerHand L20 model
# -------------------------------------------------------------------------

hand_model = HandModel(
    model_path=args.hand_model_path,
    mesh_path=args.mesh_path,
    contact_points_path=args.contact_points_path,
    penetration_points_path=args.penetration_points_path,
    n_surface_points=0,
    device=device,
    force_mesh_sdf=True
)

print('hand_model_path', args.hand_model_path)
print('mesh_path', args.mesh_path)
print('contact_points_path', args.contact_points_path)
print('penetration_points_path', args.penetration_points_path)
print('linker_z_fix_deg', args.linker_z_fix_deg)
print('n_dofs', hand_model.n_dofs)
print('joint_names', hand_model.joints_names)

if hand_model.n_dofs <= 0:
    raise RuntimeError('LinkerHand L20 n_dofs <= 0, URDF joints were not parsed.')

# -------------------------------------------------------------------------
# prepare object model
# -------------------------------------------------------------------------

object_model = ObjectModel(
    data_root_path='../data/meshdata',
    batch_size_each=args.batch_size,
    # num_samples=500,
    device=device
)
object_model.initialize(args.object_code_list)

# -------------------------------------------------------------------------
# initialize grasps
# -------------------------------------------------------------------------

initialize_convex_hull(hand_model, object_model, args)

print('n_contact_candidates', hand_model.n_contact_candidates)
print('total batch size', total_batch_size)

if hand_model.n_contact_candidates <= 0:
    raise RuntimeError('No contact candidates loaded for LinkerHand L20.')

hand_pose_st = hand_model.hand_pose.detach()

# -------------------------------------------------------------------------
# optimizer
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# logger
# -------------------------------------------------------------------------

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

logger = Logger(
    log_dir=os.path.join('../data/experiments', args.name, 'logs'),
    **logger_config
)

with open(os.path.join('../data/experiments', args.name, 'output.txt'), 'w') as f:
    f.write(str(args) + '\n')

# -------------------------------------------------------------------------
# optimize
# -------------------------------------------------------------------------

weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_spen=args.w_spen,
    w_joints=args.w_joints,
)

energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(
    hand_model,
    object_model,
    verbose=True,
    **weight_dict
)

energy.sum().backward(retain_graph=True)
logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, 0, show=False)

for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
    optimizer.try_step()

    optimizer.zero_grad()

    new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(
        hand_model,
        object_model,
        verbose=True,
        **weight_dict
    )

    new_energy.sum().backward(retain_graph=True)

    with torch.no_grad():
        accept, t = optimizer.accept_step(energy, new_energy)

        energy[accept] = new_energy[accept]
        E_dis[accept] = new_E_dis[accept]
        E_fc[accept] = new_E_fc[accept]
        E_pen[accept] = new_E_pen[accept]
        E_spen[accept] = new_E_spen[accept]
        E_joints[accept] = new_E_joints[accept]

        logger.log(
            energy,
            E_fc,
            E_dis,
            E_pen,
            E_spen,
            E_joints,
            step,
            show=False
        )
# -------------------------------------------------------------------------
# save results
# -------------------------------------------------------------------------

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = hand_model.joints_names

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'results'))
except FileNotFoundError:
    pass

result_path = os.path.join('../data/experiments', args.name, 'results')
os.makedirs(result_path, exist_ok=True)

for i in range(len(args.object_code_list)):
    data_list = []

    for j in range(args.batch_size):
        idx = i * args.batch_size + j
        scale = object_model.object_scale_tensor[i][j].item()

        # ================== Final optimized pose ==================
        hand_pose = hand_model.hand_pose[idx].detach().cpu()
        qpos = dict(zip(joint_names, hand_pose[9:].tolist()))

        rot = robust_compute_rotation_matrix_from_ortho6d(
            hand_pose[3:9].unsqueeze(0)
        )[0]

        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos.update(dict(zip(rot_names, euler)))
        qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))

        # 💡 核心奥义：因为 URDF 已经完美对齐，此时 L20 执行位姿 == ShadowHand 语义位姿
        qpos_shadow_semantic = qpos.copy()

        # ======================= Initial pose =======================
        hand_pose_init = hand_pose_st[idx].detach().cpu()
        qpos_st = dict(zip(joint_names, hand_pose_init[9:].tolist()))

        rot_st = robust_compute_rotation_matrix_from_ortho6d(
            hand_pose_init[3:9].unsqueeze(0)
        )[0]

        euler_st = transforms3d.euler.mat2euler(rot_st, axes='sxyz')
        qpos_st.update(dict(zip(rot_names, euler_st)))
        qpos_st.update(dict(zip(translation_names, hand_pose_init[:3].tolist())))

        # 同样直接拷贝即可
        qpos_st_shadow_semantic = qpos_st.copy()

        # ====================== Append Data ======================
        data_list.append(dict(
            scale=scale,
            qpos=qpos,
            qpos_st=qpos_st,
            qpos_shadow_semantic=qpos_shadow_semantic,
            qpos_st_shadow_semantic=qpos_st_shadow_semantic,
            energy=energy[idx].item(),
            E_fc=E_fc[idx].item(),
            E_dis=E_dis[idx].item(),
            E_pen=E_pen[idx].item(),
            E_spen=E_spen[idx].item(),
            E_joints=E_joints[idx].item(),
        ))

    np.save(
        os.path.join(result_path, args.object_code_list[i] + '.npy'),
        data_list,
        allow_pickle=True
    )

print(f"🎉 成功保存所有抓取数据到: {result_path}")