from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import threading
import logging
import requests
import h5py

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === File Configuration ===
MODEL_FILENAME   = "autism_detection_model_(9.4).h5"
YOLO_CFG         = "yolov3.cfg"
YOLO_WEIGHTS     = "yolov3.weights"
COCO_NAMES       = "coco.names"

# === Direct Download URLs ===
# — Replace MODEL_URL with a working direct link to your .h5
MODEL_URL        = "https://drive.usercontent.google.com/download?id=1XRZQfgJuOCGvM3vU-1wVpKaf4aP2LxvF&export=download&authuser=0&confirm=t&uuid=2f2da46b-7f74-4274-a4cd-1832c2d4cdc5&at=AN8xHoqztFmgDsR8nJY7GIyBGoEe:1749874549886"
YOLO_CFG_URL     = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
COCO_NAMES_URL   = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Flask App ===
app = Flask(__name__)
CORS(app)

model            = None
net              = None
output_layers    = None
classes          = None
model_lock       = threading.Lock()
model_load_error = None

# === Helpers to Download and Clean Files ===
def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)

def download_file(url, target):
    logger.info(f"Downloading {target} from {url}")
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code == 200:
        with open(target, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        logger.info(f"{target} downloaded")
    else:
        logger.error(f"Failed to download {target} (status {resp.status_code})")

def prepare_yolo():
    # remove any old files
    for f in (YOLO_CFG, YOLO_WEIGHTS, COCO_NAMES):
        delete_if_exists(f)
    # redownload fresh
    download_file(YOLO_CFG_URL, YOLO_CFG)
    download_file(YOLO_WEIGHTS_URL, YOLO_WEIGHTS)
    download_file(COCO_NAMES_URL, COCO_NAMES)

def prepare_model():
    delete_if_exists(MODEL_FILENAME)
    download_file(MODEL_URL, MODEL_FILENAME)

def is_valid_h5(path):
    try:
        with h5py.File(path, 'r'): return True
    except: return False

# === Load both models once, thread‑safely ===
def load_everything():
    global model, net, output_layers, classes, model_load_error
    with model_lock:
        if model and net:
            return

        # prepare files
        if not os.path.exists(MODEL_FILENAME) or not is_valid_h5(MODEL_FILENAME):
            prepare_model()
        prepare_yolo()

        # load autism model
        try:
            model = load_model(MODEL_FILENAME)
            logger.info("Autism model loaded")
        except Exception as e:
            model_load_error = f"Autism model load error: {e}"
            logger.error(model_load_error)
            return

        # load YOLO
        try:
            net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
            with open(COCO_NAMES) as f:
                classes = [l.strip() for l in f]
            layers = net.getLayerNames()
            output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            logger.info("YOLO model loaded")
        except Exception as e:
            model_load_error = f"YOLO load error: {e}"
            logger.error(model_load_error)

# === Inference Helpers ===
def contains_human(img):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = np.argmax(scores)
            if scores[cid] > 0.5 and classes[cid] == "person":
                return True
    return False

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Invalid image"}
    if not contains_human(img):
        return {"error": "No person detected"}
    pil = load_img(img_path, target_size=(299, 299))
    arr = img_to_array(pil)
    arr = np.expand_dims(arr, 0)
    return {"image": preprocess_input(arr)}

def predict_autism(img_path):
    prep = preprocess(img_path)
    if "error" in prep:
        return prep
    try:
        scores = model.predict(prep["image"])[0]
        label = ["Autism", "Normal"][np.argmax(scores)]
        return {"prediction": label}
    except Exception as e:
        return {"error": f"Inference failed: {e}"}

# === Routes ===
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(error="No file"), 400
    f = request.files["file"]
    tmp = "temp.jpg"
    f.save(tmp)
    try:
        load_everything()
        if model_load_error:
            raise RuntimeError(model_load_error)
        res = predict_autism(tmp)
        status = 200 if "prediction" in res else 400
        return jsonify(res), status
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        if os.path.exists(tmp): os.remove(tmp)

@app.route("/health")
def health():
    load_everything()
    ok = model and net
    return jsonify(model_loaded=bool(model),
                   yolo_loaded=bool(net),
                   error=model_load_error), (200 if ok else 503)


