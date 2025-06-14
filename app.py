from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading
import logging
import requests
import h5py
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Logging Configuration ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app")

# === Flask App Initialization ===
app = Flask(__name__)
CORS(app)

# === Global Variables ===
model = None
model_lock = threading.Lock()
model_load_error = None
model_filename = 'autism_detection_model_(9.4).h5'
drive_url = 'https://drive.usercontent.google.com/download?id=1XRZQfgJuOCGvM3vU-1wVpKaf4aP2LxvF&export=download'

# === Model Download Function ===
def download_model_from_drive(url, dest_path):
    try:
        logger.info("Downloading model from Google Drive...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Model downloaded successfully.")
            return True
        else:
            logger.error(f"Failed to download model. Status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Exception while downloading model: {e}")
        return False

# === Model Loader ===
def load_model_in_background():
    global model, model_load_error
    with model_lock:
        if model is None and model_load_error is None:
            model_path = os.path.abspath(model_filename)
            logger.info(f"Attempting to load model from: {model_path}")

            if not os.path.exists(model_filename) or os.path.getsize(model_filename) < 100000:
                logger.warning("Model not found or too small. Attempting to download...")
                if not download_model_from_drive(drive_url, model_filename):
                    model_load_error = "Failed to download model from Google Drive."
                    return

            try:
                file_size = os.path.getsize(model_filename)
                logger.info(f"Model file size: {file_size} bytes")
                if file_size < 100000:
                    logger.error(f"Model file too small ({file_size} bytes), likely corrupted")
                    model_load_error = f"Model file too small ({file_size} bytes)"
                    return

                with h5py.File(model_filename, 'r') as f:
                    logger.info("HDF5 file validated")

                model = load_model(model_filename)
                logger.info("Model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                model_load_error = f"Failed to load model: {str(e)}"

# === Lazy Model Loader ===
def load_model_if_needed():
    with model_lock:
        if model is None and model_load_error:
            raise RuntimeError(model_load_error)
        if model is None:
            load_model_in_background()
        return model

# === Image Blurriness Detection ===
def is_image_blurry(image, threshold=100):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold
    except Exception as e:
        logger.error(f"Blurriness check failed: {e}")
        return True

# === Image Preprocessing ===
def preprocess_image(image_path, target_size=(299, 299)):
    try:
        original = cv2.imread(image_path)
        if original is None:
            logger.error(f"Could not read image at {image_path}")
            return {"error": "Invalid image or unreadable file."}
        if is_image_blurry(original):
            logger.warning(f"Image too blurry: {image_path}")
            return {"error": "The image is too blurry."}
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return {"image": image}
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return {"error": f"Error preprocessing image: {str(e)}"}

# === Prediction Logic ===
def predict_image(model, image_path):
    result = preprocess_image(image_path)
    if "error" in result:
        return result
    try:
        image = result["image"]
        prediction = model.predict(image)[0]
        predicted_class = np.argmax(prediction)
        class_labels = ['Autism', 'Normal']
        return {"prediction": class_labels[predicted_class]}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

# === Routes ===

@app.route('/health', methods=['GET'])
def health():
    with model_lock:
        status = "healthy" if model is not None else "unhealthy - model not loaded"
        error = None if model is not None else model_load_error
    return jsonify({"status": status, "error": error}), 200 if model is not None else 503

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join(os.getcwd(), 'temp_image.jpg')
    try:
        file.save(image_path)
        logger.info(f"Image saved at: {image_path}")

        model = load_model_if_needed()
        result = predict_image(model, image_path)
        logger.info(f"Prediction: {result}")

        if "error" in result:
            return jsonify(result), 400
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

    finally:
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                logger.info("Temporary image removed")
            except Exception as e:
                logger.error(f"Failed to delete temporary image: {e}")

# === Run App ===
if __name__ == '__main__':
    logger.warning("Running in development mode. Use Gunicorn in production.")
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Server running on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)
