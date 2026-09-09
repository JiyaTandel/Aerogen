import json
import glob

# Find reconstruction.json from ODM's OpenSfM step
files = glob.glob("**/reconstruction.json", recursive=True)
if not files:
    print("Could not find reconstruction.json!")
    exit()

print(f"Found: {files[0]}")
with open(files[0], "r") as f:
    data = json.load(f)

rec = data[0] if isinstance(data, list) else data
shots = rec.get("shots", {})

print(f"Total reconstructed shots: {len(shots)}")
print("\nSample image keys:")
for name in list(shots.keys())[:5]:
    print(f"  - {name}")
    