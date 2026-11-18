"""
Last modified date: 2026.05.02
Description: Entry of the program, generate LinkerHand L20 grasps (Clean Version)
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
import json
from pathlib import Path

from utils.hand_model_v2 import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.logger import Logger
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.grasp_results import save_scale_groups
from utils.contact_library import load_library,attach_library,world_contact_normals,normal_alignment
from utils.root_repulsion import filtered_self_repulsion,ignored_pairs


# prepare arguments
parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--object_code_list', default=
    [
        'core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03'
    ], nargs='+')
parser.add_argument('--name', default='l20_exp', type=str)
parser.add_argument('--n_contact', default=4 , type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--n_iter', default=6000, type=int)
parser.add_argument('--object-scales', '--object_scales', nargs='+', type=float,
                    help='Explicit uniform mesh scales, balanced across each object batch')
parser.add_argument('--hull-inflation', choices=['paper', 'scale-aware'], default='paper',
                    help='Initialization hull offset: paper=0.2 m; scale-aware=small-object L20 experiment')
parser.add_argument('--contact-library', type=str, help='Explicit versioned point/frame directory; default preserves old library')
parser.add_argument('--w-normal', type=float, default=0., help='Additional hand/object antiparallel-normal loss; default disabled')
parser.add_argument('--root-contact-policy',choices=['all','angle-only','all-roots-angle-only'],default='all')
parser.add_argument('--penetration-geometry',choices=['legacy-convex','raw-mesh'],default='legacy-convex')
parser.add_argument('--contact-sampling',choices=['global','grouped4','grouped5'],default='global')

# hyper parameters (** Magic, don't touch! **)
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
parser.add_argument('--distance_lower', default=0.05, type=float)
parser.add_argument('--distance_upper', default=0.10, type=float)
parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
parser.add_argument('--theta_upper', default=math.pi / 6, type=float)

# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

args = parser.parse_args()
if args.contact_sampling!='global':args.n_contact=4 if args.contact_sampling=='grouped4' else 5
if not np.isfinite(args.w_normal) or args.w_normal<0:parser.error('Invalid normal weight')
if args.w_normal and not args.contact_library:parser.error('--w-normal requires --contact-library')
library_path=Path(args.contact_library).resolve() if args.contact_library else None
if library_path:
    load_library(library_path)
    if (Path('../data/experiments')/args.name).exists():parser.error('New-library experiment output must be new')


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# prepare models
total_batch_size = len(args.object_code_list) * args.batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)

# 💡 微创手术 1：修改加载路径为 L20 的 URDF，并开启 SDF 支持
hand_model = HandModel(
    model_path='mjcf/linkerhand_l20_right.urdf', 
    mesh_path='mjcf/meshes',
    contact_points_path=str(library_path/'contact_points.json') if library_path else 'mjcf/contact_points.json',
    penetration_points_path='mjcf/penetration_points.json',
    device=device,
    force_mesh_sdf=args.penetration_geometry=='raw-mesh',penetration_geometry=args.penetration_geometry
)
hand_model.contact_sampling=args.contact_sampling
print('penetration geometry',args.penetration_geometry,'contact sampling',args.contact_sampling,'slots',args.n_contact,flush=True)
contact_provenance=attach_library(hand_model,library_path) if library_path else None

object_model = ObjectModel(
    data_root_path='../data/meshdata',
    batch_size_each=args.batch_size,
    num_samples=2000, 
    device=device
)
object_model.initialize(args.object_code_list, scales=args.object_scales)

initialize_convex_hull(hand_model, object_model, args)

print('n_contact_candidates', hand_model.n_contact_candidates)
print('total batch size', total_batch_size)
hand_pose_st = hand_model.hand_pose.detach().clone()
print('independent optimization joints', len(hand_model.joint_constraints.independent_names))

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


# log settings
with open(os.path.join('../data/experiments', args.name, 'output.txt'), 'w') as f:
    f.write(str(args) + '\n')
if library_path:
    run=Path('../data/experiments')/args.name
    shutil.copytree(library_path,run/'contact_library')
    (run/'penetration_geometry.json').write_text(json.dumps(hand_model.geometry_audit,indent=2)+'\n')
    np.save(run/'object_surface_points.npy',object_model.surface_points_tensor.detach().cpu().numpy())
    contact_provenance.update(w_normal=args.w_normal,seed=args.seed,n_contact=args.n_contact,root_contact_policy=args.root_contact_policy,
        penetration_geometry=args.penetration_geometry,contact_sampling=args.contact_sampling,algorithm_version='l20-generation-v2',
        contact_selection=('global random slots, legacy behavior' if args.contact_sampling=='global' else args.contact_sampling+': disjoint finger pools, unique IDs, preserved at initialization and every proposal'),
        disabled_repulsion_link_pairs=[sorted(x) for x in sorted(ignored_pairs(args.root_contact_policy),key=lambda x:sorted(x))],
        code_sha256={str(p.resolve()):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('utils/contact_library.py'),Path('utils/energy.py'),Path('utils/optimizer.py'),Path('utils/root_repulsion.py'),Path('utils/hand_model_v2.py'),Path('utils/contact_sampling.py'),Path('../scripts/l20_root_collision_policy.py')]})
    (run/'generation_contact_model.json').write_text(json.dumps(contact_provenance,indent=2)+'\n')


# optimize
weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_spen=args.w_spen,
    w_joints=args.w_joints,
)
def evaluate():
    terms=list(cal_energy(hand_model, object_model, verbose=True, **weight_dict))
    if args.root_contact_policy!='all':
        terms[4]=filtered_self_repulsion(hand_model,args.root_contact_policy)
        terms[0]=terms[1]+args.w_dis*terms[2]+args.w_pen*terms[3]+args.w_spen*terms[4]+args.w_joints*terms[5]
    alignment=torch.zeros_like(terms[0])
    if library_path:
        _,object_n=object_model.cal_distance(hand_model.contact_points)
        alignment=normal_alignment(world_contact_normals(hand_model),object_n)
        terms[0]=terms[0]+args.w_normal*alignment
    return terms,alignment

(energy, E_fc, E_dis, E_pen, E_spen, E_joints),E_normal = evaluate()

energy.sum().backward(retain_graph=True)
logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, 0, show=False)
if library_path:logger.writer.add_scalar('Energy/normal',E_normal.mean(),0)

for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
    s = optimizer.try_step()

    optimizer.zero_grad()
    (new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints),new_E_normal = evaluate()

    new_energy.sum().backward(retain_graph=True)

    with torch.no_grad():
        accept, t = optimizer.accept_step(energy, new_energy)

        energy[accept] = new_energy[accept]
        E_dis[accept] = new_E_dis[accept]
        E_fc[accept] = new_E_fc[accept]
        E_pen[accept] = new_E_pen[accept]
        E_spen[accept] = new_E_spen[accept]
        E_joints[accept] = new_E_joints[accept]
        E_normal[accept] = new_E_normal[accept]

        logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, show=False)
        if library_path:logger.writer.add_scalar('Energy/normal',E_normal.mean(),step)


# save results
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

# 💡 微创手术 2：动态获取关节名，不再硬编码 ShadowHand 的 22 个关节
joint_names = hand_model.joints_names 

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
        
        # 💡 微创手术 3：兼容动态解析出的 L20 关节数量
        qpos = dict(zip(joint_names, hand_pose[9:9+len(joint_names)].tolist())) 
        
        # 🔥 修复 1: 强制转换为 numpy 数组，以适配 transforms3d 的输入要求
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0].numpy()
        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos.update(dict(zip(rot_names, euler)))
        qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
        
        # 🔥 修复 2: 隔离变量名，防止 hand_pose, rot, euler 相互覆盖污染数据
        hand_pose_st_idx = hand_pose_st[idx].detach().cpu()
        qpos_st = dict(zip(joint_names, hand_pose_st_idx[9:9+len(joint_names)].tolist()))
        
        # 🔥 修复 3: 这里也必须转换为 numpy 数组
        rot_st = robust_compute_rotation_matrix_from_ortho6d(hand_pose_st_idx[3:9].unsqueeze(0))[0].numpy()
        euler_st = transforms3d.euler.mat2euler(rot_st, axes='sxyz')
        qpos_st.update(dict(zip(rot_names, euler_st)))
        qpos_st.update(dict(zip(translation_names, hand_pose_st_idx[:3].tolist())))
        
        data_list.append(dict(
            scale=scale,
            source_index=j,
            qpos=qpos,
            qpos_st=qpos_st,
            energy=energy[idx].item(),
            E_fc=E_fc[idx].item(),
            E_dis=E_dis[idx].item(),
            E_pen=E_pen[idx].item(),
            E_spen=E_spen[idx].item(),
            E_joints=E_joints[idx].item(),
            root_contact_policy=args.root_contact_policy,
            penetration_geometry=args.penetration_geometry,contact_sampling=args.contact_sampling,
            **(dict(E_normal=E_normal[idx].item(),normal_weight=args.w_normal,
                contact_library_manifest_sha256=contact_provenance['manifest_sha256'],
                contact_point_indices=hand_model.contact_point_indices[idx].detach().cpu().tolist(),
                contact_stable_ids=hand_model.library_stable_ids[hand_model.contact_point_indices[idx]].detach().cpu().tolist(),
                contact_normals_world=world_contact_normals(hand_model)[idx].detach().cpu().tolist()) if library_path else {}),
        ))
    np.save(os.path.join(result_path, args.object_code_list[i] + '.npy'), data_list, allow_pickle=True)
    save_scale_groups(result_path, args.object_code_list[i], data_list,
                      object_model.object_mesh_list[i].vertices,
                      os.path.join('../data/meshdata', args.object_code_list[i], 'coacd/decomposed.obj'))

print(f"🎉 成功保存所有抓取数据到: {result_path}")
