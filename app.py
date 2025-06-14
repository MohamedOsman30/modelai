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

# === Configuration ===
MODEL_FILENAME = "autism_detection_model_(9.4).h5"
YOLO_CFG = "yolov3.cfg"
YOLO_WEIGHTS = "yolov3.weights"
COCO_NAMES = "coco.names"

MODEL_URL = "https://drive.usercontent.google.com/download?id=1XRZQfgJuOCGvM3vU-1wVpKaf4aP2LxvF&export=download&authuser=0&confirm=t&uuid=e543c47a-ecf4-4dc0-b728-ad887595bfe6&at=AN8xHorwP4Bq1QmL7PBu--JlDN0G:1749882792114"
YOLO_CFG_URL = "https://drive.usercontent.google.com/download?id=1Kw5wpmAc-OlzIeD6StseoRwulEzYn7S6&export=download&authuser=0&confirm=t&uuid=d636bfc9-ab48-49b0-bf2e-a888c9dce94b&at=AN8xHoqEv5HxfF0-SX_fEQq-t-Dy:1749882860744"
YOLO_WEIGHTS_URL = "https://drive.usercontent.google.com/download?id=1ze3bnBoeR0lXwzqVSqn4G9moPB4Xtdy5&export=download&authuser=0&confirm=t&uuid=cc3d9eb0-5a98-4afd-96c7-c80fc37d6550&at=AN8xHopkaJnQW6EgDml1cYrzy7-J:1749882902561"
COCO_NAMES_URL = "https://drive.usercontent.google.com/download?id=1tXqtfva92vsKbBgbVV4GxKb9NUZ-JM4P&export=download&authuser=0&confirm=t&uuid=b8c6878f-298a-4947-afc2-026d06e69fbc&at=AN8xHooPiqFR7sbjVslN8NR96qC-:1749882928916"

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Flask App ===
app = Flask(__name__)
CORS(app)

model = None
net = None
output_layers = None
classes = None
model_lock = threading.Lock()
model_load_error = None

# === File Download ===
def download_file(url, filename):
    logger.info(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"{filename} downloaded.")
        else:
            logger.error(f"Failed to download {filename}, status: {response.status_code}")
    except Exception as e:
        logger.error(f"Exception downloading {filename}: {e}")

def download_yolo_files():
    if not os.path.exists(YOLO_CFG):
        download_file(YOLO_CFG_URL, YOLO_CFG)
    if not os.path.exists(YOLO_WEIGHTS):
        download_file(YOLO_WEIGHTS_URL, YOLO_WEIGHTS)
    if not os.path.exists(COCO_NAMES):
        download_file(COCO_NAMES_URL, COCO_NAMES)

def download_model():
    download_file(MODEL_URL, MODEL_FILENAME)

def is_valid_h5_file(filepath):
    try:
        with h5py.File(filepath, 'r'):
            return True
    except:
        return False

def load_model_and_yolo():
    global model, net, output_layers, classes, model_load_error
    with model_lock:
        if model is not None and net is not None:
            return

        # Download missing files
        if not os.path.exists(MODEL_FILENAME) or not is_valid_h5_file(MODEL_FILENAME):
            download_model()

        download_yolo_files()

        # Load Keras model
        try:
            model = load_model(MODEL_FILENAME)
            logger.info("Autism model loaded.")
        except Exception as e:
            model_load_error = f"Model load failed: {e}"
            return

        # Load YOLO
        try:
            net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
            with open(COCO_NAMES, "r") as f:
                classes = [line.strip() for line in f.readlines()]
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            logger.info("YOLO model loaded.")
        except Exception as e:
            model_load_error = f"YOLO load failed: {e}"

# === Utility Functions ===
def is_image_blurry(image, threshold=100):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold
    except Exception as e:
        logger.error(f"Blurry check failed: {e}")
        return True

def contains_human(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                return True
    return False

def preprocess_image(image_path):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        return {"error": "Invalid image"}

    

    if not contains_human(image_cv):
        return {"error": "Image does not contain a human"}

    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return {"image": image}

def predict_image(image_path):
    result = preprocess_image(image_path)
    if "error" in result:
        return result
    try:
        preds = model.predict(result["image"])[0]
        label = ["Autism", "Normal"][np.argmax(preds)]
        return {"prediction": label}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# === Routes ===
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_path = "temp.jpg"
    file.save(image_path)

    try:
        load_model_and_yolo()
        if model is None or net is None:
            raise RuntimeError("Model or YOLO not loaded")
        result = predict_image(image_path)
        return jsonify(result), 200 if "prediction" in result else 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "model_loaded": model is not None,
        "yolo_loaded": net is not None,
        "error": model_load_error
    }), 200 if model and net else 503

