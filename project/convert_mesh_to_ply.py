import os
import glob
import trimesh

# Search recursively for any odm_textured_model.obj or any .obj
obj_candidates = glob.glob("**/*.obj", recursive=True)

if not obj_candidates:
    raise FileNotFoundError("No .obj file found in this folder or subfolders!")

# Pick the first matching mesh
mesh_path = obj_candidates[0]
print(f"Found mesh at: {mesh_path}")

output_dir = "outputs/enhanced_v1"
output_ply = os.path.join(output_dir, "point_cloud.ply")
os.makedirs(output_dir, exist_ok=True)

print("Loading mesh...")
mesh = trimesh.load_mesh(mesh_path)

print("Sampling 1,000,000 points from mesh...")
points, _ = trimesh.sample.sample_surface(mesh, count=1_000_000)

print(f"Exporting to {output_ply}...")
pcd = trimesh.PointCloud(vertices=points)
pcd.export(output_ply)

print(f"Done! Created: {output_ply} with {len(points):,} points.")