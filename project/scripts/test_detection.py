import os
import glob
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
# COCO dynamic classes: 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
MOVING_CLASSES = [0, 2, 3, 5, 7]

IMAGES_DIR = os.path.join("odm_project_clean", "images")
images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.[jJ][pP][gG]")))

print(f"Scanning all {len(images)} images with conf=0.10 and imgsz=1280...")

found_any = False
for img_path in images:
    base = os.path.basename(img_path)
    res = model(img_path, classes=MOVING_CLASSES, imgsz=1280, conf=0.10, verbose=False)[0]
    
    count = len(res.boxes) if res.boxes is not None else 0
    if count > 0:
        names = [model.names[int(c)] for c in res.boxes.cls]
        print(f"  --> {base}: detected {count} object(s) -> {names}")
        found_any = True
    else:
        print(f"{base}: 0")

if not found_any:
    print("\nNo moving objects detected even at conf=0.10.")