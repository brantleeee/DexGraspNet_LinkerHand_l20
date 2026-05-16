import os
import shutil
import trimesh
from tqdm import tqdm

# 1. 设置路径
source_dir = '/root/DexGraspNet/data/ycb_small/ycb'    # 你的 ycb_small 目录
target_dir = '/root/DexGraspNet/data/meshdata'         # DexGraspNet 官方要求的读取目录

os.makedirs(target_dir, exist_ok=True)

# 获取所有物体文件夹（排除 .urdf 和 .xml 文件）
object_folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

print(f"找到 {len(object_folders)} 个物体，开始转换为 DexGraspNet 格式...")

for obj_name in tqdm(object_folders):
    src_obj_dir = os.path.join(source_dir, obj_name)
    target_obj_dir = os.path.join(target_dir, obj_name)
    target_coacd_dir = os.path.join(target_obj_dir, 'coacd')
    
    os.makedirs(target_coacd_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 步骤 A: 处理视觉网格 (Visual Mesh)
    # 逻辑: 将 textured.obj 直接重命名复制为 coacd.obj
    # ---------------------------------------------------------
    src_visual = os.path.join(src_obj_dir, 'textured.obj')
    target_visual = os.path.join(target_obj_dir, 'coacd.obj')
    if os.path.exists(src_visual):
        shutil.copy(src_visual, target_visual)
    
    # ---------------------------------------------------------
    # 步骤 B: 处理碰撞网格 (Collision Mesh)
    # 逻辑: 找到所有 textured_coacd_X.stl，合并它们，导出为 decomposed.obj
    # ---------------------------------------------------------
    stl_files = [f for f in os.listdir(src_obj_dir) if f.startswith('textured_coacd_') and f.endswith('.stl')]
    
    if stl_files:
        meshes = []
        for stl in stl_files:
            meshes.append(trimesh.load(os.path.join(src_obj_dir, stl), force='mesh'))
        
        # 将多个凸块合并为一个 Trimesh 对象
        merged_mesh = trimesh.util.concatenate(meshes)
        
        # 导出为 DexGraspNet 要求的格式
        merged_mesh.export(os.path.join(target_coacd_dir, 'decomposed.obj'))

print("\n🎉 转换完成！所有数据已就绪！")