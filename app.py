from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import logging
import threading
import h5py
import requests

# --- Configuration ---
MODEL_FILENAME = "autism_detection_model_(9.4).h5"
DRIVE_DIRECT_DOWNLOAD = "https://drive.usercontent.google.com/download?id=1XRZQfgJuOCGvM3vU-1wVpKaf4aP2LxvF&export=download&authuser=0&confirm=t&uuid=2f2da46b-7f74-4274-a4cd-1832c2d4cdc5&at=AN8xHoqztFmgDsR8nJY7GIyBGoEe:1749874549886"

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask app ---
app = Flask(__name__)
CORS(app)

model = None
model_lock = threading.Lock()
model_load_error = None

def download_model_from_drive():
    logger.info("Attempting to download model from Google Drive...")
    try:
        response = requests.get(DRIVE_DIRECT_DOWNLOAD, stream=True, timeout=60)
        if response.status_code == 200:
            with open(MODEL_FILENAME, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Model downloaded successfully.")
            return True
        else:
            logger.error(f"Failed to download model. Status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Exception during model download: {e}")
        return False

def is_valid_h5_file(filepath):
    try:
        with h5py.File(filepath, 'r'):
            return True
    except:
        return False

def load_model_in_background():
    global model, model_load_error
    with model_lock:
        if model is not None or model_load_error:
            return
        logger.info(f"Loading model from: {MODEL_FILENAME}")
        if not os.path.exists(MODEL_FILENAME) or os.path.getsize(MODEL_FILENAME) < 1_000_000:
            logger.warning("Model file missing or corrupted, attempting to download...")
            if not download_model_from_drive():
                model_load_error = "Model download failed"
                return
        if not is_valid_h5_file(MODEL_FILENAME):
            model_load_error = "Downloaded file is not a valid HDF5 model."
            return
        try:
            model = load_model(MODEL_FILENAME)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            model_load_error = str(e)

# Start loading model in background
threading.Thread(target=load_model_in_background, daemon=True).start()

@app.route("/health", methods=["GET"])
def health():
    with model_lock:
        return jsonify({
            "status": "ready" if model else "not ready",
            "error": model_load_error
        }), 200 if model else 503

def is_image_blurry(image, threshold=100):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold
    except Exception as e:
        logger.error(f"Blurry check failed: {e}")
        return True

def preprocess_image(image_path):
    original_image = cv2.imread(image_path)
    if original_image is None:
        return {"error": "Invalid image"}
    if is_image_blurry(original_image):
        return {"error": "Image too blurry"}
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return {"image": image}

def predict_image(model, image_path):
    result = preprocess_image(image_path)
    if "error" in result:
        return result
    try:
        preds = model.predict(result["image"])[0]
        label = ["Autism", "Normal"][np.argmax(preds)]
        return {"prediction": label}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    image_path = "temp.jpg"
    file.save(image_path)
    try:
        load_model_in_background()
        if not model:
            raise RuntimeError("Model not ready")
        result = predict_image(model, image_path)
        return jsonify(result), 200 if "prediction" in result else 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
