import trimesh
import json
import os

# 确保这里的路径指向你刚建好的 L6 描述文件夹
meshes_dir = "l6_hand_description/meshes"

# 你的五个指尖连杆名称
distal_links = [
    "thumb_distal", 
    "index_distal", 
    "middle_distal", 
    "ring_distal", 
    "pinky_distal"
]

contact_points = {}

for link_name in distal_links:
    mesh_path = os.path.join(meshes_dir, f"{link_name}.STL")
    if os.path.exists(mesh_path):
        mesh = trimesh.load(mesh_path)
        # 在每个指尖表面随机撒 50 个点作为候选接触点
        points, _ = trimesh.sample.sample_surface(mesh, 50)
        contact_points[link_name] = points.tolist()
        print(f"Generated 50 points for {link_name}")
    else:
        print(f"Error: {mesh_path} not found!")

# 保存为 json
output_path = "l6_hand_description/contact_points.json"
with open(output_path, "w") as f:
    json.dump(contact_points, f)
print(f"Saved contact points to {output_path}")