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
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go
from tqdm import tqdm

from utils.hand_model import HandModel
from utils.object_model import ObjectModel

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认指向批量生成脚本保存的文件夹
    parser.add_argument('--result_path', type=str, default='../data/l20_graspdata')
    parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    parser.add_argument('--all', action='store_true', help='Visualize all objects in the result_path')
    parser.add_argument('--num_vis', type=int, default=1, help='每个物体生成多少个抓取的 HTML (默认 1)')
    args = parser.parse_args()

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

        # 💡 核心修复 1：读取数据后，立即按总能量从小到大排序
        # 确保排在最前面的永远是物理上最完美、穿模最小的最优抓取
        all_data = np.load(npy_path, allow_pickle=True)
        all_data = sorted(all_data, key=lambda x: x['energy'])
        
        # 防止要求的渲染数量大于实际生成的数量
        render_count = min(args.num_vis, len(all_data))

        # 3. 初始化 object model (每个物体重新初始化，防止网格重叠)
        object_model = ObjectModel(
            data_root_path='../data/meshdata',
            batch_size_each=1,
            num_samples=2000, 
            device=device
        )
        object_model.initialize([obj_code])

        for k in range(render_count):
            data_dict = all_data[k]
            qpos = data_dict['qpos']
            
            # 解析末端姿态
            rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
            rot = rot[:, :2].T.ravel().tolist()
            # 使用安全的 get() 避免键名不匹配导致的报错
            hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos.get(name, 0.0) for name in joint_names], dtype=torch.float, device=device)
            
            # 解析初始姿态 (st)
            if 'qpos_st' in data_dict:
                qpos_st = data_dict['qpos_st']
                rot_st = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
                rot_st = rot_st[:, :2].T.ravel().tolist()
                hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot_st + [qpos_st.get(name, 0.0) for name in joint_names], dtype=torch.float, device=device)

            object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

            # =========================================================================
            # 4. 渲染与导出
            # =========================================================================
            if 'qpos_st' in data_dict:
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
                result = f'Rank: #{k+1} | Scale: {scale}x | E_fc: {E_fc} | E_dis: {E_dis} | E_pen: {E_pen}'
                fig.add_annotation(text=result, x=0.5, y=0.05, xref='paper', yref='paper', showarrow=False, font=dict(size=14, color="red"))
                
            fig.update_layout(scene_aspectmode='data', title=f"L20 Grasp - {obj_code} (Rank #{k+1})")
            
            # 导出 HTML (保存在 result_path 中，名字带有排名)
            html_path = os.path.join(args.result_path, f"{obj_code}_vis_rank{k+1}.html")
            fig.write_html(html_path)

    print(f"\n🎉 批量可视化完成！HTML 文件已保存在: {os.path.abspath(args.result_path)}")