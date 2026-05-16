import xml.etree.ElementTree as ET
import trimesh
import trimesh.transformations as tf
import numpy as np
import json
import os

# ================= 👑 核心算法：最远点采样 (FPS) =================
def farthest_point_sampling(points, num_points):
    """确保装甲点均匀贴合表面，拒绝扎堆和漏洞"""
    if len(points) == 0: return np.empty((0, 3))
    if num_points >= len(points): return points

    selected_points = np.zeros((num_points, 3))
    distances = np.ones(len(points)) * np.inf
    farthest_idx = np.random.randint(0, len(points))

    for i in range(num_points):
        selected_points[i] = points[farthest_idx]
        dist_to_new_point = np.linalg.norm(points - selected_points[i], axis=1)
        distances = np.minimum(distances, dist_to_new_point)
        farthest_idx = np.argmax(distances)

    return selected_points
# ===================================================================

# ================= 路径配置 =================
base_dir = "l20_right_description"
urdf_path = os.path.join(base_dir, "linkerhand_l20_right.urdf")
meshes_dir = os.path.join(base_dir, "meshes")
output_path = os.path.join(base_dir, "penetration_points.json")
# ========================================================

print("🛡️ 正在进行 L20 右手深度 URDF 解析与运动学建树...")

tree = ET.parse(urdf_path)
root = tree.getroot()

# 👑 运动学树：解析关节信息，用于追踪连杆的绝对位置
joint_tree = {}
for joint in root.findall('joint'):
    parent = joint.find('parent').attrib['link']
    child = joint.find('child').attrib['link']
    origin = joint.find('origin')
    xyz = [float(x) for x in origin.attrib['xyz'].split()] if origin is not None and 'xyz' in origin.attrib else [0, 0, 0]
    rpy = [float(x) for x in origin.attrib['rpy'].split()] if origin is not None and 'rpy' in origin.attrib else [0, 0, 0]
    joint_tree[child] = {'parent': parent, 'xyz': xyz, 'rpy': rpy}

def get_transform_to_base(target_link):
    """计算从目标连杆到 base_link 的空间变换矩阵"""
    mat = np.eye(4)
    curr = target_link
    while curr in joint_tree:
        node = joint_tree[curr]
        local_mat = tf.euler_matrix(*node['rpy'], 'sxyz')
        local_mat[:3, 3] = node['xyz']
        mat = local_mat @ mat
        curr = node['parent']
    return mat

# ---------------- 阶段 1：解析 URDF 提取正确姿态的 Mesh ----------------
link_meshes = {}
areas = {}
total_area = 0.0

for link in root.findall('link'):
    link_name = link.attrib['name']

    targets = link.findall('collision')
    if not targets:
        targets = link.findall('visual')
        
    combined_vertices = []
    combined_faces = []
    vertex_offset = 0
    
    for target in targets:
        geom = target.find('geometry')
        if geom is None: continue
        mesh_tag = geom.find('mesh')
        if mesh_tag is None: continue
            
        filename = mesh_tag.attrib['filename']
        mesh_basename = os.path.basename(filename)
        mesh_path = os.path.join(meshes_dir, mesh_basename)
        
        if not os.path.exists(mesh_path):
            continue
            
        scale = [1.0, 1.0, 1.0]
        if 'scale' in mesh_tag.attrib:
            scale = [float(x) for x in mesh_tag.attrib['scale'].split()]
            
        origin = target.find('origin')
        xyz, rpy = [0, 0, 0], [0, 0, 0]
        if origin is not None:
            if 'xyz' in origin.attrib: xyz = [float(x) for x in origin.attrib['xyz'].split()]
            if 'rpy' in origin.attrib: rpy = [float(x) for x in origin.attrib['rpy'].split()]
                
        try:
            tm_mesh = trimesh.load(mesh_path, force='mesh')
            tm_mesh.vertices[:, 0] *= scale[0]
            tm_mesh.vertices[:, 1] *= scale[1]
            tm_mesh.vertices[:, 2] *= scale[2]
            
            # 将 Mesh 变换到连杆的局部原点坐标系下
            matrix = tf.euler_matrix(rpy[0], rpy[1], rpy[2], 'sxyz')
            matrix[:3, 3] = xyz
            tm_mesh.apply_transform(matrix)
            
            combined_vertices.append(tm_mesh.vertices)
            combined_faces.append(tm_mesh.faces + vertex_offset)
            vertex_offset += len(tm_mesh.vertices)
        except Exception as e:
            pass

    if combined_vertices:
        v = np.vstack(combined_vertices)
        f = np.vstack(combined_faces)
        merged_mesh = trimesh.Trimesh(vertices=v, faces=f)
        link_meshes[link_name] = merged_mesh
        areas[link_name] = merged_mesh.area
        total_area += merged_mesh.area

# ---------------- 阶段 2：按面积比例 + 空间裁剪 + FPS 撒点 ----------------
# 👑 核心更新：新增密度控制参数，防止 DexGraspNet 训练/生成时显存溢出
TOTAL_TARGET_POINTS = 300     # 从 15000 下调至 3000，适配 bs=128
MIN_POINTS_PER_LINK = 10       # 保证微小连杆的最小特征点数
MAX_POINTS_PER_LINK = 40      # 限制大连杆点数上限，防止算力倾斜
OVERSAMPLE_RATE = 5           # 超采样倍率从 30 下调至 10，提升预处理速度

penetration_points = {}
total_generated = 0

print("\n" + "="*50)
print("🛡️ 正在使用 [面积加权] 与 [空间裁剪] 重铸半嵌入式装甲 (已启用点云数量控制)...")
print("="*50 + "\n")

# 提取 base_link 作为判断半嵌入式结构的物理边界
base_mesh = link_meshes.get("base_link")

# 👑 核心更新：把四个手指的掌骨也加入“半嵌入式”切割列表
embedded_links = [
    "thumb_metacarpals", "thumb_metacarpals_base1", "thumb_metacarpals_base2", 
    "index_metacarpals", "middle_metacarpals", "ring_metacarpals", "pinky_metacarpals"
]

# =========================================================================
# 👑 新增：穿模点“黑名单”，被列入此名单的连杆将被完全跳过采集
# =========================================================================
excluded_links = [
    "pinky_metacarpals", 
    "ring_metacarpals", 
    "middle_metacarpals", 
    "index_metacarpals", 
    "thumb_metacarpals_base1", 
    "thumb_metacarpals_base2"
]

for link_name, mesh in link_meshes.items():
    # 🎯 拦截机制：如果在黑名单中，直接赋空并跳过
    if link_name in excluded_links:
        print(f"🚫 {link_name.ljust(25)}: 在黑名单中，强制跳过采集防穿模点！")
        penetration_points[link_name] = []
        continue

    # 依据面积计算初始目标点数
    raw_target_num = int((areas[link_name] / total_area) * TOTAL_TARGET_POINTS)
    
    # 应用上下限截断逻辑
    target_num = max(MIN_POINTS_PER_LINK, min(raw_target_num, MAX_POINTS_PER_LINK))
        
    try:
        # 高倍率超采样 (已降为10倍率优化时间)
        raw_points, face_indices = trimesh.sample.sample_surface(mesh, target_num * OVERSAMPLE_RATE)
        normals = mesh.face_normals[face_indices]
        shrunk_points = raw_points - normals * 0.0002
        
        # 👑 核心修复：执行空间物理裁剪 (切除长在肉里的那一半)
        if link_name in embedded_links and base_mesh is not None:
            mat_to_base = get_transform_to_base(link_name)
            
            # 将当前点阵变换到 base_link 坐标系
            pts_h = np.hstack((shrunk_points, np.ones((len(shrunk_points), 1))))
            pts_in_base = (mat_to_base @ pts_h.T).T[:, :3]
            
            # 使用 contains 检测哪些点深深陷入了 base_link 内部
            try:
                inside_mask = base_mesh.contains(pts_in_base)
                shrunk_points = shrunk_points[~inside_mask]
                print(f"   ✂️ 几何切割: {link_name} 已剔除 {inside_mask.sum()} 个陷入 base_link 内部的废点！")
            except Exception as e:
                print(f"   ⚠️ 切割检测失败，保持原样: {e}")
                
        # 保证切割后仍有足够的点进行 FPS
        if len(shrunk_points) < target_num and len(shrunk_points) > 0:
            actual_target = len(shrunk_points)
        else:
            actual_target = target_num

        # 对切割后的点云执行 FPS
        fps_points = farthest_point_sampling(shrunk_points, actual_target)
        
        penetration_points[link_name] = fps_points.tolist()
        total_generated += len(fps_points)
        print(f"✅ {link_name.ljust(25)}: 生成 {len(fps_points)} 个暴露面防穿模点")
        
    except Exception as e:
        print(f"❌ 生成 {link_name} 点阵时出错: {e}")

with open(output_path, "w") as f:
    json.dump(penetration_points, f, indent=4)

print("-" * 50)
print(f"🎉 动态网格装甲重铸完成！总计 {total_generated} 个暴露点已保存至: {output_path}")