import os
import requests

# -------------------------------
# Helper: Download from URL
# -------------------------------
def download_url(url, dest_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/octet-stream,application/octet-stream,*/*"
    }
    with requests.get(url, stream=True, headers=headers, allow_redirects=True) as r:
        r.raise_for_status()
        content_type = r.headers.get('Content-Type', '')
        content_length = r.headers.get('Content-Length', 'Unknown')
        print(f"Status Code: {r.status_code}")
        print(f"Content-Type: {content_type}")
        print(f"Content-Length: {content_length} bytes")
        if 'text/html' in content_type.lower():
            print("❌ Error: Received HTML instead of the file!")
            raise ValueError(f"Download failed for {url}: Got HTML content")
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    file_size = os.path.getsize(dest_path)
    print(f"Download complete. File size: {file_size} bytes")
    return file_size

# -------------------------------
# YOLOv3 Weights & Config
# -------------------------------
print("📦 Downloading YOLOv3 files...")
os.makedirs("yolo", exist_ok=True)

yolo_files = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "yolov3.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights"
}

for fname, url in yolo_files.items():
    path = os.path.join("yolo", fname)
    if not os.path.exists(path):
        print(f"⬇️  Downloading {fname}...")
        try:
            download_url(url, path)
        except Exception as e:
            print(f"❌ Failed to download {fname}: {e}")
    else:
        print(f"✅ {fname} already exists, skipping.")

# -------------------------------
# Autism Detection Model
# -------------------------------
print("\n🧠 Downloading autism_model_9.4.h5...")
model_name = "autism_model_9.4.h5"
direct_url = "https://media.githubusercontent.com/media/MohamedOsman30/modelai/refs/heads/master/autism_detection_model_(9.4).h5?download=true"  # Replace with your GitHub release URL

if not os.path.exists(model_name):
    try:
        print(f"⬇️  Downloading from: {direct_url}")
        file_size = download_url(direct_url, model_name)
        
        # Verify the file
        expected_size = 256827392  # 244 MB in bytes
        if abs(file_size - expected_size) > 1000000:  # Allow 1 MB tolerance
            print(f"❌ Warning: File size {file_size} bytes does not match expected {expected_size} bytes!")
        
        with open(model_name, 'rb') as f:
            signature = f.read(4)
            if signature == b'\x89HDF':
                print("✅ File is a valid HDF5 file.")
            else:
                print("❌ File is not a valid HDF5 file.")
                raise ValueError("Downloaded file is not a valid HDF5 file.")
    except Exception as e:
        print("❌ Failed to download the model:")
        print(e)
else:
    print("✅ Model already exists, skipping.")
