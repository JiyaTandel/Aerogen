import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

print("Loading YOLOv8 segmentation model...")
model = YOLO('yolov8n-seg.pt')

# COCO dynamic classes: 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
MOVING_CLASSES = [0, 2, 3, 5, 7]

IMAGES_DIR = os.path.join("odm_project_clean", "images")
MASKS_DIR = os.path.join("odm_project_clean", "masks")
os.makedirs(MASKS_DIR, exist_ok=True)

valid_exts = {".jpg", ".jpeg"}
image_files = [
    os.path.join(IMAGES_DIR, f)
    for f in os.listdir(IMAGES_DIR)
    if os.path.splitext(f)[1].lower() in valid_exts
] if os.path.exists(IMAGES_DIR) else []

image_files.sort()

print(f"\nProcessing {len(image_files)} frames with imgsz=1280, conf=0.10...\n")

masked_count = 0
for idx, f in enumerate(image_files, 1):
    base_name = os.path.basename(f)
    img = cv2.imread(f)
    if img is None:
        continue
        
    h, w = img.shape[:2]
    # 255 = keep (white canvas)
    mask = np.full((h, w), 255, dtype=np.uint8)

    results = model(f, classes=MOVING_CLASSES, imgsz=1280, conf=0.10, verbose=False)
    result = results[0]

    found_objects = 0
    if result.masks is not None:
        raw_masks = result.masks.data.cpu().numpy()
        for m in raw_masks:
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            # 0 = ignore (black silhouette over moving objects)
            mask[m_resized > 0.5] = 0
            found_objects += 1

    if found_objects > 0:
        masked_count += 1

    out_path = os.path.join(MASKS_DIR, base_name)
    cv2.imwrite(out_path, mask)
    print(f"[{idx}/{len(image_files)}] {base_name}: masked {found_objects} dynamic object(s)")

print(f"\nDone! Masks saved to '{MASKS_DIR}'.")
print(f"Frames with masked objects: {masked_count}/{len(image_files)}")