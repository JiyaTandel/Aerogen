import cv2, glob, shutil, os

def blur_score(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()

os.makedirs("odm_project_clean/images", exist_ok=True)
scores = []
for f in glob.glob("odm_project/images/*.jpg"):
    s = blur_score(f)
    scores.append((f, s))

scores.sort(key=lambda x: x[1])
print("Lowest scores (blurriest):", scores[:5])
print("Highest scores (sharpest):", scores[-5:])

# pick a threshold by eyeballing the printed scores — usually bottom 10-15% are unusable
threshold = scores[int(len(scores)*0.12)][1]
kept = 0
for f, s in scores:
    if s >= threshold:
        shutil.copy(f, f"odm_project_clean/images/{os.path.basename(f)}")
        kept += 1
print(f"Kept {kept}/{len(scores)} frames")