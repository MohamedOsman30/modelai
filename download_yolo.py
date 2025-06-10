import os
import requests
import sys

# -------------------------------
# Helper: Download from URL
# -------------------------------
def download_url(url, dest_path):
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)



# -------------------------------
# YOLOv3 Weights & Config
# -------------------------------
print("📦 Downloading YOLOv3 files...")
os.makedirs("yolo", exist_ok=True)

yolo_files = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights"
}

for fname, url in yolo_files.items():
    path = os.path.join("yolo", fname)
    if not os.path.exists(path):
        print(f"⬇️  Downloading {fname}...")
        download_url(url, path)
    else:
        print(f"✅ {fname} already exists, skipping.")


