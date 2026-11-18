"""
Modified for LinkerHand L20 Visualization
- Kept original ObjectModel logic
- Dynamic joint_names fetching
"""

import os
import sys

# 兼容路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
sys.path.append(parent_dir)

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.grasp_results import verify_mesh_identity

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    parser.add_argument('--num', type=int, default=None, help='Explicit candidate index; default selects the best candidate passing energy thresholds')
    parser.add_argument('--show_initial', action='store_true', help='Overlay the initial pose in translucent pink')
    parser.add_argument('--result_path', type=str, default='../data/experiments/l20_test_run/results')
    args = parser.parse_args()
    verify_mesh_identity(os.path.join(args.result_path,args.object_code+'.npy'),
                         os.path.join('../data/meshdata',args.object_code,'coacd/decomposed.obj'))

    device = 'cpu'

    # =========================================================================
    # 1. 先初始化 hand model，动态获取关节名
    # =========================================================================
    hand_model = HandModel(
        model_path='mjcf/linkerhand_l20_right.urdf',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        n_surface_points=0,
        device=device,
        force_mesh_sdf=True
    )
    joint_names = hand_model.joints_names

    # =========================================================================
    # 2. 读取实验结果 (使用动态 joint_names)
    # =========================================================================
    all_data = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)
    if args.num is None:
        valid_indices = [i for i, grasp in enumerate(all_data)
                         if grasp['E_fc'] < 0.3 and grasp['E_dis'] < 0.005 and grasp['E_pen'] < 0.001]
        candidates = valid_indices or list(range(len(all_data)))
        selected_index = min(candidates, key=lambda i: all_data[i]['energy'])
        print(f'Candidates passing contact/penetration thresholds: {len(valid_indices)}/{len(all_data)}')
        if not valid_indices:
            print('WARNING: No candidate passes the thresholds; displaying the lowest-energy candidate.')
    else:
        selected_index = args.num
    data_dict = all_data[selected_index]
    print(f'Selected candidate {selected_index}: E_fc={data_dict["E_fc"]:.6g}, E_dis={data_dict["E_dis"]:.6g} m, E_pen={data_dict["E_pen"]:.6g}')
    qpos = data_dict['qpos']
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
    
    if args.show_initial and 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot + [qpos_st[name] for name in joint_names], dtype=torch.float, device=device)

    # =========================================================================
    # 3. object model (严格保留附件原代码的逻辑)
    # =========================================================================
    object_model = ObjectModel(
        data_root_path='../data/meshdata',
        batch_size_each=1,
        num_samples=0,
        device=device
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    # =========================================================================
    # 4. visualize
    # =========================================================================
    if args.show_initial and 'qpos_st' in data_dict:
        hand_model.set_parameters(hand_pose_st.unsqueeze(0))
        hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.25, color='lightpink', with_contact_points=False)
    else:
        hand_st_plotly = []
        
    hand_model.set_parameters(hand_pose.unsqueeze(0))
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
    
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    
    if 'energy' in data_dict:
        energy = data_dict['energy']
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        E_spen = round(data_dict['E_spen'], 5)
        E_joints = round(data_dict['E_joints'], 5)
        passed = data_dict['E_fc'] < 0.3 and data_dict['E_dis'] < 0.005 and data_dict['E_pen'] < 0.001
        result = f'Index {selected_index} | Scale {data_dict["scale"]:.4g} | Contact thresholds: {"PASS" if passed else "FAIL"} | E_fc {E_fc} | E_dis {E_dis} m | E_pen {E_pen}'
        fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
        
    fig.update_layout(scene_aspectmode='data')
    # Look from the object side, so the wrist shell does not hide a small object.
    wrist_position = np.array([qpos[name] for name in translation_names])
    if np.linalg.norm(wrist_position) > 1e-6:
        eye = -1.8*wrist_position/np.linalg.norm(wrist_position) + np.array([0.,0.,.3])
        fig.update_layout(scene_camera_eye=dict(zip(('x','y','z'),eye)))
    
    # 导出 HTML
    suffix = 'best' if args.num is None else str(args.num)
    html_path = os.path.join(args.result_path, f"{args.object_code}_vis_{suffix}.html")
    fig.write_html(html_path)
    print(f"✅ Visualization saved to {html_path}")
