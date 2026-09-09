import glob
import json
import os
import numpy as np
import torch
import trimesh
from PIL import Image
from transformers import pipeline

# 1. Paths & Reconstruction
rec_files = glob.glob("**/reconstruction.json", recursive=True)
if not rec_files:
    raise FileNotFoundError("Could not find reconstruction.json!")

REC_PATH = rec_files[0]
OUTPUT_PLY = "outputs/final/confidence_model.ply"
os.makedirs("outputs/final", exist_ok=True)

with open(REC_PATH, "r") as f:
    rec_data = json.load(f)

rec = rec_data[0] if isinstance(rec_data, list) else rec_data
shots = rec.get("shots", {})
cameras = rec.get("cameras", {})
all_points = rec.get("points", {})

shot_keys = list(shots.keys())
total_shots = len(shot_keys)

# Pick 3 spread-out camera angles across the sequence
target_indices = [0, total_shots // 2, max(0, total_shots - 1)]
target_frames = [shot_keys[i] for i in target_indices]
print(f"Targeting 3 distributed frames: {target_frames}")

# 2. Init Depth Anything V2
print("Loading Depth Anything V2...")
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=-1,
)

all_predicted_pts = []

for idx, frame_name in enumerate(target_frames, start=1):
    print(f"\n[{idx}/3] Processing {frame_name}...")
    shot = shots[frame_name]
    cam = cameras[shot["camera"]]

    img_candidates = glob.glob(f"**/{frame_name}", recursive=True)
    if not img_candidates:
        print(f"  Warning: Skipping {frame_name}, file not found.")
        continue
    
    img_path = img_candidates[0]
    raw_img = Image.open(img_path).convert("RGB")
    raw_img.thumbnail((1024, 1024))
    w_orig, h_orig = raw_img.size

    # Depth inference
    depth_output = pipe(raw_img)["depth"]
    depth_map = np.array(depth_output).astype(np.float32)

    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    depth_rel = 1.0 / (depth_norm + 0.05)

    # Intrinsics
    max_dim = max(w_orig, h_orig)
    fx = cam.get("focal", 0.85) * max_dim
    fy = fx
    cx = w_orig / 2.0 + cam.get("c_x", 0.0) * max_dim
    cy = h_orig / 2.0 + cam.get("c_y", 0.0) * max_dim

    # Extrinsics
    rvec = np.array(shot["rotation"])
    angle = np.linalg.norm(rvec)
    if angle < 1e-6:
        R = np.eye(3)
    else:
        k = rvec / angle
        K_mat = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * (K_mat @ K_mat)

    t = np.array(shot["translation"]).reshape(3, 1)

    # Scale alignment with SfM points
    tracked_z = []
    for pt_id in shot.get("vertices", []):
        if pt_id in all_points:
            pt_w = np.array(all_points[pt_id]["coordinates"]).reshape(3, 1)
            pt_cam = R @ pt_w + t
            if pt_cam[2] > 0:
                tracked_z.append(pt_cam[2, 0])

    median_z = np.median(tracked_z) if len(tracked_z) > 0 else 25.0
    depth_scaled = depth_rel * (median_z / np.median(depth_rel))

    # Mask a localized patch
    h_d, w_d = depth_scaled.shape
    mask = np.zeros((h_d, w_d), dtype=bool)
    h_crop, w_crop = int(h_d * 0.35), int(w_d * 0.35)
    mask[
        (h_d // 2 - h_crop // 2):(h_d // 2 + h_crop // 2),
        (w_d // 2 - w_crop // 2):(w_d // 2 + w_crop // 2),
    ] = True

    u_grid, v_grid = np.meshgrid(np.arange(w_orig), np.arange(h_orig))
    u_down = (u_grid[::4, ::4]).flatten()
    v_down = (v_grid[::4, ::4]).flatten()

    scale_x = w_d / w_orig
    scale_y = h_d / h_orig
    u_d_idx = np.clip((u_down * scale_x).astype(int), 0, w_d - 1)
    v_d_idx = np.clip((v_down * scale_y).astype(int), 0, h_d - 1)

    valid_mask = mask[v_d_idx, u_d_idx]
    u_sel = u_down[valid_mask]
    v_sel = v_down[valid_mask]
    z_sel = depth_scaled[v_d_idx[valid_mask], u_d_idx[valid_mask]]

    x_cam = (u_sel - cx) * z_sel / fx
    y_cam = (v_sel - cy) * z_sel / fy
    pts_cam = np.vstack([x_cam, y_cam, z_sel])

    pts_world = (R.T @ (pts_cam - t)).T
    all_predicted_pts.append(pts_world)
    print(f"  Generated {len(pts_world):,} patch points.")

# Combine all patches
combined_predicted = np.vstack(all_predicted_pts)

# Save intermediate depth points
pred_cloud = trimesh.PointCloud(vertices=combined_predicted)
pred_cloud.export("depth_filled_points.ply")
print(f"\nSaved {len(combined_predicted):,} total predicted points to depth_filled_points.ply")

# Merge with ODM base cloud
print("Merging base ODM cloud (Green) with 3 confidence regions (Orange)...")
base_cloud = trimesh.load("outputs/enhanced_v1/point_cloud.ply")
base_pts = np.asarray(base_cloud.vertices)

green = np.array([0, 215, 0, 255], dtype=np.uint8)
base_colors = np.tile(green, (len(base_pts), 1))

orange = np.array([255, 140, 0, 255], dtype=np.uint8)
pred_colors = np.tile(orange, (len(combined_predicted), 1))

merged_pts = np.vstack([base_pts, combined_predicted])
merged_colors = np.vstack([base_colors, pred_colors])

final_cloud = trimesh.PointCloud(vertices=merged_pts, colors=merged_colors)
final_cloud.export(OUTPUT_PLY)

print(f"\nSUCCESS: Multi-region confidence model saved to: {OUTPUT_PLY}")
print(f"Total points: {len(merged_pts):,} (ODM Green: {len(base_pts):,}, Predicted Orange: {len(combined_predicted):,})")