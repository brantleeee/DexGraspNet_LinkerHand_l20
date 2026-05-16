import os
import sys

# 兼容各种路径运行
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go

from utils.hand_model import HandModel
from utils.object_model import ObjectModel

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 1. 默认参数改为你的测试环境
    parser.add_argument('--object_code', type=str, default='adjustable_wrench')
    parser.add_argument('--num', type=int, default=0) # 改为 0，配合后面的自动排序，永远看最优抓取
    # 👑 修改点 1: 默认读取你刚刚跑出来的 L20 实验结果文件夹
    parser.add_argument('--result_path', type=str, default='../data/experiments/l20_05011643/results')
    args = parser.parse_args()

    device = 'cpu'

    # =========================================================================
    # 2. 逻辑倒置：必须先初始化 HandModel，以获取 L20 真实的关节名称！
    # =========================================================================
    # 👑 修改点 2: 全面替换为 L20 的文件路径
    hand_model = HandModel(
        urdf_path='l20_right_description/linkerhand_l20_right.urdf',
        contact_points_path='l20_right_description/contact_points.json', 
        penetration_points_path='l20_right_description/penetration_points.json',
        device=device
    )
    
    # 动态获取 L20 的真实关节名，彻底消灭硬编码！
    joint_names = hand_model.chain.get_joint_parameter_names()

    # =========================================================================
    # 3. 读取结果数据并解析张量
    # =========================================================================
    npy_path = os.path.join(args.result_path, args.object_code + '.npy')
    if not os.path.exists(npy_path):
        print(f"❌ 找不到结果文件: {npy_path}")
        sys.exit()

    # 智能优化：按总能量从小到大排序，确保 --num 0 永远是最优抓取
    all_data = np.load(npy_path, allow_pickle=True)
    all_data = sorted(all_data, key=lambda x: x['energy'])
    data_dict = all_data[args.num]

    qpos = data_dict['qpos']
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    
    # 使用安全的 get() 方法，防止意外的 KeyError
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos.get(name, 0.0) for name in joint_names], dtype=torch.float, device=device)
    
    if 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot + [qpos_st.get(name, 0.0) for name in joint_names], dtype=torch.float, device=device)

    # =========================================================================
    # 4. 初始化 ObjectModel
    # =========================================================================
    object_model = ObjectModel(
        data_root_path='../data/meshdata',
        batch_size_each=1,
        num_samples=2000, 
        device=device
    )
    # 兼容传入列表的初始化方式
    object_model.initialize([args.object_code])
    # 找到这行原始代码，将它注释掉！
    # object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)
    
    # 👑 替换为强行指定的可视化尺寸：
    # object_model.object_scale_tensor = torch.tensor([[1.0]], dtype=torch.float, device=device)

    # 恢复读取真实训练时的物理缩放尺寸，拒绝视觉欺骗！
    if 'scale' in data_dict:
        object_model.object_scale_tensor = torch.tensor([[data_dict['scale']]], dtype=torch.float, device=device)
    else:
        object_model.object_scale_tensor = torch.tensor([[1.0]], dtype=torch.float, device=device)
    


    # =========================================================================
    # 5. 渲染与 HTML 导出
    # =========================================================================
    if 'qpos_st' in data_dict:
        hand_model.set_parameters(hand_pose_st.unsqueeze(0))
        # 初始化的“虚影”手用粉色半透明表示
        hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.3, color='lightpink', visual=True)
    else:
        hand_st_plotly = []
        
    hand_model.set_parameters(hand_pose.unsqueeze(0))
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', visual=True)
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
    
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    
    if 'energy' in data_dict:
        scale = round(data_dict['scale'], 2)
        energy = round(data_dict['energy'], 3)
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        
        result = f'Object: {args.object_code} | Total Energy: {energy} | E_fc: {E_fc} | E_dis: {E_dis} | E_pen: {E_pen}'
        fig.add_annotation(text=result, x=0.5, y=0.05, xref='paper', yref='paper', showarrow=False, font=dict(size=14, color="red"))
        
    # 👑 修改点 3: 标题修改为 L20
    fig.update_layout(scene_aspectmode='data', title="L20 Grasp Visualizer")
    
    # 核心修改：禁止直接 show()，改为输出 HTML 网页
    output_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{args.object_code}_result_vis.html")
    fig.write_html(output_html)
    print(f"✅ 可视化渲染成功！文件已保存至: {output_html}")