import trimesh
import json
import os
import numpy as np

# ================= 👑 核心算法：最远点采样 (FPS) =================
def farthest_point_sampling(points, num_points):
    if len(points) == 0:
        return np.empty((0, 3))
    if num_points >= len(points):
        return points

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
meshes_dir = "l20_right_description/meshes"
output_path = "l20_right_description/contact_points.json"
# ========================================================

# ================= 核心控制面板 =================
config = {
    # 👑 手掌全覆盖模式 (修正名称为 base_link)
    # "base_link": {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 500, "w_margin_ratio": 0.04, "l_margin_top": 0.10, "l_margin_bottom": 0.10, "f_ratio": 0.17},
    "base_link": {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 500, "w_margin_ratio": 0.05, "l_margin_top": 0.1, "l_margin_bottom": 0.3, "f_ratio": 0.35},
    # 四指 - 近端 (Proximal)
    "index_proximal": {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.05, "l_margin_top": 0.10, "l_margin_bottom": 0.4, "f_ratio": 0.15},
    "middle_proximal":{"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.05, "l_margin_top": 0.10, "l_margin_bottom": 0.4, "f_ratio": 0.15},
    "ring_proximal":  {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.05, "l_margin_top": 0.10, "l_margin_bottom": 0.4, "f_ratio": 0.15},
    "pinky_proximal": {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.05, "l_margin_top": 0.10, "l_margin_bottom": 0.4, "f_ratio": 0.15},

    # 四指 - 中端 (Middle) - L20新增
    "index_middle":  {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.05, "l_margin_top": 0.15, "l_margin_bottom": 0.1, "f_ratio": 0.15},
    "middle_middle": {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.05, "l_margin_top": 0.15, "l_margin_bottom": 0.1, "f_ratio": 0.15},
    "ring_middle":   {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.05, "l_margin_top": 0.15, "l_margin_bottom": 0.1, "f_ratio": 0.15},
    "pinky_middle":  {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.05, "l_margin_top": 0.15, "l_margin_bottom": 0.1, "f_ratio": 0.15},

    # 四指 - 远端 (Distal)
    "index_distal":   {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 150, "w_margin_ratio": 0.05, "l_margin_top": 0.0, "l_margin_bottom": 0.4, "f_ratio": 0.6},
    "middle_distal":  {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 150, "w_margin_ratio": 0.05, "l_margin_top": 0.0, "l_margin_bottom": 0.4, "f_ratio": 0.6},
    "ring_distal":    {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 150, "w_margin_ratio": 0.05, "l_margin_top": 0.0, "l_margin_bottom": 0.4, "f_ratio": 0.6},
    "pinky_distal":   {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 150, "w_margin_ratio": 0.05, "l_margin_top": 0.0, "l_margin_bottom": 0.4, "f_ratio": 0.6},

    # 👑 大拇指曲面包络模式 (已加入 L20 的 proximal 环节)
    "thumb_metacarpals": {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.15, "l_margin_top": 0.3, "l_margin_bottom": 0.3, "f_ratio": 0.55},
    "thumb_proximal":    {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.05, "l_margin_top": 0.2, "l_margin_bottom": 0.35, "f_ratio": 0.55},
    "thumb_distal":      {"face_axis": 0, "face_dir": 1, "width_axis": 1, "length_axis": 2, "target_num": 100, "w_margin_ratio": 0.02, "l_margin_top": 0.00, "l_margin_bottom": 0.40, "f_ratio": 0.55},
}
# ===================================================================

OUTWARD_OFFSET = 0.0002
GRID_RES = 0.001

contact_points = {}

for link_name, params in config.items():
    mesh_path = os.path.join(meshes_dir, f"{link_name}.STL")
    if not os.path.exists(mesh_path):
        print(f"⚠️ {link_name}：找不到 STL 文件，跳过。")
        continue

    mesh = trimesh.load(mesh_path)
    raw_points, face_indices = trimesh.sample.sample_surface(mesh, 100000) 
    raw_normals = mesh.face_normals[face_indices]

    min_vals = np.min(raw_points, axis=0)
    max_vals = np.max(raw_points, axis=0)

    f_axis, f_dir = params["face_axis"], params["face_dir"]
    w_axis, l_axis = params["width_axis"], params["length_axis"]
    f_ratio, target = params["f_ratio"], params["target_num"]

    l_min, l_max = min_vals[l_axis], max_vals[l_axis]
    l_margin_bottom_val = (l_max - l_min) * params["l_margin_bottom"]
    l_margin_top_val = (l_max - l_min) * params["l_margin_top"]

    w_min, w_max = min_vals[w_axis], max_vals[w_axis]
    w_margin = (w_max - w_min) * params["w_margin_ratio"]

    f_min, f_max = min_vals[f_axis], max_vals[f_axis]
    f_range = f_max - f_min

    candidate_points = []
    candidate_normals = []
    
    for idx, p in enumerate(raw_points):
        if not ((l_min + l_margin_bottom_val) < p[l_axis] < (l_max - l_margin_top_val)): continue
        if not ((w_min + w_margin) < p[w_axis] < (w_max - w_margin)): continue
        
        if f_dir == 1 and p[f_axis] <= f_max - f_ratio * f_range: continue
        if f_dir == -1 and p[f_axis] >= f_min + f_ratio * f_range: continue

        if raw_normals[idx][f_axis] * f_dir < 0.2: continue
        
        candidate_points.append(p)
        candidate_normals.append(raw_normals[idx])

    depth_map = {}
    for i, p in enumerate(candidate_points):
        grid_l = int(np.round(p[l_axis] / GRID_RES))
        grid_w = int(np.round(p[w_axis] / GRID_RES))
        key = (grid_l, grid_w)

        if key not in depth_map:
            depth_map[key] = p
        else:
            existing_p = depth_map[key]
            if f_dir == 1 and p[f_axis] > existing_p[f_axis]:
                depth_map[key] = p
            elif f_dir == -1 and p[f_axis] < existing_p[f_axis]:
                depth_map[key] = p

    filtered_points = []
    for p in depth_map.values():
        p_pushed = p.copy()
        p_pushed[f_axis] += OUTWARD_OFFSET * f_dir
        filtered_points.append(p_pushed)

    filtered_points = np.array(filtered_points)

    if len(filtered_points) >= target:
        fps_points = farthest_point_sampling(filtered_points, target)
        contact_points[link_name] = fps_points.tolist()
        print(f"✅ {link_name}：成功使用 FPS 生成均匀网格阵列！(提取 {target} 个点)")
    elif len(filtered_points) > 0:
        contact_points[link_name] = filtered_points.tolist()
        print(f"⚠️ {link_name}：可用点数({len(filtered_points)})少于目标数，已保留全部可用点。")
    else:
        contact_points[link_name] = []
        print(f"❌ {link_name}：未提取到点！")

with open(output_path, "w") as f:
    json.dump(contact_points, f, indent=4)

print(f"\n🎉 FPS 点阵化右手接触点生成完毕！已保存至 {output_path}")