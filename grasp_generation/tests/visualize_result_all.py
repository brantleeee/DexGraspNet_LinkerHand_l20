"""
Modified for LinkerHand L20 Visualization (Batch Processing Support & Smart Sorting)
"""

import os
import sys

# =========================================================================
# 兼容路径配置：无论你在哪个目录运行这个脚本，它都会自动定位到 grasp_generation
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if os.path.basename(parent_dir) != 'grasp_generation':
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.chdir(parent_dir)
sys.path.append(parent_dir)

import argparse
import html
import json
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go
from tqdm import tqdm

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.grasp_results import verify_mesh_identity

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认指向批量生成脚本保存的文件夹
    parser.add_argument('--result_path', type=str, default='../data/l20_graspdata')
    parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    parser.add_argument('--all', action='store_true', help='Visualize all objects in the result_path')
    parser.add_argument('--num_vis', type=int, default=1, help='每个物体生成多少个抓取的 HTML (默认 1)')
    parser.add_argument('--show_initial', action='store_true', help='Overlay the initial pose in translucent pink')
    parser.add_argument('--sort', choices=['index', 'energy'], default='energy', help='index preserves original candidate numbering')
    parser.add_argument('--output_dir', help='Separate HTML output directory; default: result_path')
    args = parser.parse_args()
    if args.num_vis < 0:
        parser.error('--num_vis must be nonnegative; 0 means all candidates')
    output_dir = os.path.abspath(args.output_dir or args.result_path)
    os.makedirs(output_dir, exist_ok=True)

    device = 'cpu'

    # 获取要处理的物体列表
    if args.all:
        if not os.path.exists(args.result_path):
            print(f"❌ 找不到结果目录: {args.result_path}")
            sys.exit(1)
        object_codes = [f.replace('.npy', '') for f in os.listdir(args.result_path) if f.endswith('.npy')]
        object_codes.sort()
    else:
        object_codes = [args.object_code]

    if len(object_codes) == 0:
        print("⚠️ 未找到任何 .npy 抓取结果文件。")
        sys.exit(0)

    print(f"🔍 准备可视化 {len(object_codes)} 个物体...")

    # =========================================================================
    # 1. 先初始化 hand model，动态获取关节名 (全局只初始化一次)
    # 严格对齐 main.py 中的 mjcf 路径和 SDF 强制开启
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
    # 批处理循环开始
    # =========================================================================
    for obj_code in tqdm(object_codes, desc="Visualizing Objects"):
        npy_path = os.path.join(args.result_path, obj_code + '.npy')
        if not os.path.exists(npy_path):
            print(f"⚠️ 找不到文件跳过: {npy_path}")
            continue

        # Prefer candidates passing the energy thresholds, then rank by energy.
        all_data = np.load(npy_path, allow_pickle=True)
        verify_mesh_identity(npy_path, os.path.join('../data/meshdata', obj_code, 'coacd/decomposed.obj'))
        def passes_thresholds(grasp):
            return grasp['E_fc'] < 0.3 and grasp['E_dis'] < 0.005 and grasp['E_pen'] < 0.001
        valid_count = sum(passes_thresholds(grasp) for grasp in all_data)
        print(f'{obj_code}: {valid_count}/{len(all_data)} candidates pass contact/penetration thresholds')
        indexed_data = list(enumerate(all_data))
        if args.sort == 'energy':
            indexed_data.sort(key=lambda item: (not passes_thresholds(item[1]), item[1]['energy']))
        
        # 防止要求的渲染数量大于实际生成的数量
        render_count = min(args.num_vis, len(all_data)) if args.num_vis else len(all_data)
        catalog = []

        # 3. 初始化 object model (每个物体重新初始化，防止网格重叠)
        object_model = ObjectModel(
            data_root_path='../data/meshdata',
            batch_size_each=1,
            num_samples=0,
            device=device
        )
        object_model.initialize([obj_code])

        for k in range(render_count):
            candidate_index, data_dict = indexed_data[k]
            qpos = data_dict['qpos']
            
            # 解析末端姿态
            rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
            rot = rot[:, :2].T.ravel().tolist()
            # 使用安全的 get() 避免键名不匹配导致的报错
            hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos.get(name, 0.0) for name in joint_names], dtype=torch.float, device=device)
            
            # 解析初始姿态 (st)
            if args.show_initial and 'qpos_st' in data_dict:
                qpos_st = data_dict['qpos_st']
                rot_st = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
                rot_st = rot_st[:, :2].T.ravel().tolist()
                hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot_st + [qpos_st.get(name, 0.0) for name in joint_names], dtype=torch.float, device=device)

            object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

            # =========================================================================
            # 4. 渲染与导出
            # =========================================================================
            if args.show_initial and 'qpos_st' in data_dict:
                hand_model.set_parameters(hand_pose_st.unsqueeze(0))
                # 初始化的“起手势”用粉色半透明表示
                hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.3, color='lightpink', with_contact_points=False)
            else:
                hand_st_plotly = []
                
            hand_model.set_parameters(hand_pose.unsqueeze(0))
            # 最终的“闭合势”用浅蓝色表示
            hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
            object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
            
            fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
            
            if 'energy' in data_dict:
                scale = round(data_dict['scale'], 2)
                energy = round(data_dict['energy'], 3)
                E_fc = round(data_dict['E_fc'], 3)
                E_dis = round(data_dict['E_dis'], 5)
                E_pen = round(data_dict['E_pen'], 5)
                
                # 💡 核心修复 2：在界面上显式打印出当前的 Scale 比例和能量排名
                status = 'PASS' if passes_thresholds(data_dict) else 'FAIL'
                result = f'Candidate: {candidate_index} | Geometric thresholds: {status} | Scale: {scale}x | E_fc: {E_fc} | E_dis: {E_dis} | E_pen: {E_pen}'
                fig.add_annotation(text=result, x=0.5, y=0.05, xref='paper', yref='paper', showarrow=False, font=dict(size=14, color="red"))
                
            fig.update_layout(scene_aspectmode='data', title=f"DexGraspNet L20 - {obj_code} | candidate {candidate_index} | scale {float(data_dict['scale']):.6g}")
            wrist_position = np.array([qpos[name] for name in translation_names])
            if np.linalg.norm(wrist_position) > 1e-6:
                eye = -1.8 * wrist_position / np.linalg.norm(wrist_position) + np.array([0., 0., .3])
                fig.update_layout(scene_camera_eye=dict(zip(('x', 'y', 'z'), eye)))
            
            # 导出 HTML (保存在 result_path 中，名字带有排名)
            filename = f"{obj_code}_vis_candidate_{candidate_index:03d}.html"
            html_path = os.path.join(output_dir, filename)
            fig.write_html(html_path, include_plotlyjs='directory')
            catalog.append({'candidate_index': candidate_index,
                            'source_batch_index': int(data_dict.get('source_index', candidate_index)),
                            'scale': float(data_dict['scale']), 'html': filename})

        gallery = os.path.join(output_dir, f'{obj_code}_gallery.html')
        links = '\n'.join(f'<li><a href="{html.escape(c["html"], quote=True)}" target="grasp">Candidate {c["candidate_index"]} — scale {c["scale"]:.6g}</a></li>' for c in catalog)
        first = html.escape(catalog[0]['html'], quote=True) if catalog else ''
        with open(gallery, 'w') as stream:
            stream.write('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                         '<title>DexGraspNet generated grasps</title>'
                         '<style>body{font-family:sans-serif;margin:0;display:flex;height:100vh}nav{width:280px;padding:16px;overflow:auto}li{margin:12px 0}iframe{flex:1;border:0}</style>'
                         f'<nav><h2>{html.escape(obj_code)}</h2><p>全部 {len(catalog)} 个生成姿态。编号对应原始 NPY 索引。</p>'
                         f'<p>静态生成结果；不代表动力学验证通过。</p><ul>{links}</ul></nav>'
                         f'<iframe name="grasp" src="{first}"></iframe></html>')
        with open(os.path.join(output_dir, f'{obj_code}_catalog.json'), 'w') as stream:
            json.dump({'source_file': os.path.abspath(npy_path), 'candidates': catalog}, stream, indent=2)
        print(f'Gallery: {gallery}')

    print(f"\nHTML output: {output_dir}")
