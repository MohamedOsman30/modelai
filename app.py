from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading
import logging
import cv2
import numpy as np
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
model_lock = threading.Lock()
model_load_error = None

# Load model in background
def load_model_background():
    global model, model_load_error
    with model_lock:
        model_path = "autism_detection_model_(9.4).h5"
        abs_path = os.path.abspath(model_path)
        logger.info(f"Attempting to load model from: {abs_path}")
        if not os.path.exists(model_path):
            model_load_error = f"Model file not found at {abs_path}"
            logger.error(model_load_error)
            return
        try:
            file_size = os.path.getsize(model_path)
            logger.info(f"Model file size: {file_size} bytes")
            if file_size < 1000:
                model_load_error = f"Model file too small ({file_size} bytes), likely corrupted"
                logger.error(model_load_error)
                return
            with h5py.File(model_path, 'r'):
                logger.info("Model file validated successfully.")
            model = load_model(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            model_load_error = f"Failed to load model: {str(e)}"
            logger.error(model_load_error)

# Start model loading in a background thread
threading.Thread(target=load_model_background, daemon=True).start()

# Helper: Load model if needed
def get_model():
    global model
    with model_lock:
        if model is not None:
            return model
        if model_load_error:
            raise RuntimeError(model_load_error)
        raise RuntimeError("Model is still loading.")

# Health route
@app.route("/")
@app.route("/health", methods=["GET"])
def health():
    with model_lock:
        status = "ready" if model else "loading"
        error = None if model else model_load_error
    return jsonify({"status": status, "error": error}), 200 if model else 503

# Check if image is blurry
def is_blurry(image, threshold=100):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold
    except Exception as e:
        logger.error(f"Error checking blur: {e}")
        return True

# Image preprocessing
def preprocess_image(image_path, target_size=(299, 299)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Invalid image path or unreadable image."}
        if is_blurry(img):
            return {"error": "Image is too blurry."}

        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return {"image": image}
    except Exception as e:
        return {"error": f"Preprocessing error: {str(e)}"}

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    temp_path = os.path.join(os.getcwd(), "temp_image.jpg")
    try:
        file.save(temp_path)
        logger.info(f"Image saved to {temp_path}")
        model_instance = get_model()
        preprocess_result = preprocess_image(temp_path)

        if "error" in preprocess_result:
            return jsonify(preprocess_result), 400

        image = preprocess_result["image"]
        prediction = model_instance.predict(image)[0]
        class_idx = np.argmax(prediction)
        labels = ["Autism", "Normal"]
        predicted_label = labels[class_idx]
        return jsonify({"prediction": predicted_label}), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Temp file deleted.")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")

# Run locally only (not used by Docker/Gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
