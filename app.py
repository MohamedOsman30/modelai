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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Log startup details
port = os.getenv('PORT', '5000')
logger.info(f"Starting Flask app on host 0.0.0.0 and port {port}")

# Model initialization
model = None
model_lock = threading.Lock()

def load_model_in_background():
    global model
    with model_lock:
        if model is None:
            model_path = 'autism_detection_model_(9.4).h5'
            abs_model_path = os.path.abspath(model_path)
            logger.info(f"Attempting to load model from: {abs_model_path}")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {abs_model_path}")
                raise FileNotFoundError(f"Model file not found at {abs_model_path}")
            try:
                # Validate HDF5 file and log file size
                file_size = os.path.getsize(model_path)
                logger.info(f"Model file size: {file_size} bytes")
                with h5py.File(model_path, 'r') as f:
                    logger.info(f"Validated HDF5 file at {abs_model_path}")
                model = load_model(model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

# Start model loading in a separate thread
threading.Thread(target=load_model_in_background, daemon=True).start()

def load_model_if_needed():
    with model_lock:
        if model is None:
            logger.info("Waiting for model to load...")
            load_model_in_background()
        return model

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Function to check if the image is blurry
def is_image_blurry(image, threshold=100):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance_of_laplacian < threshold
    except Exception as e:
        logger.error(f"Error checking image blurriness: {e}")
        return True  # Treat as blurry if error occurs

# Preprocessing the image
def preprocess_image(image_path, target_size=(299, 299)):
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.error(f"Failed to read image at {image_path}")
            return {"error": "Invalid image path or image could not be read."}

        if is_image_blurry(original_image):
            logger.warning(f"Image at {image_path} is too blurry")
            return {"error": "The image is too blurry."}

        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return {"image": image}
    except Exception as e:
        logger.error(f"Error preprocessing image at {image_path}: {e}")
        return {"error": f"Error preprocessing image: {str(e)}"}

# Prediction function
def predict_image(model, image_path):
    preprocess_result = preprocess_image(image_path)
    if "error" in preprocess_result:
        return preprocess_result

    try:
        image = preprocess_result["image"]
        prediction = model.predict(image)[0]
        predicted_class = np.argmax(prediction)
        class_labels = ['Autism', 'Normal']
        predicted_label = class_labels[predicted_class]
        return {"prediction": predicted_label}
    except Exception as e:
        logger.error(f"Error during prediction for {image_path}: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file in request")
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join(os.getcwd(), 'temp_image.jpg')
    try:
        file.save(image_path)
        logger.info(f"Image saved temporarily at: {image_path}")

        model = load_model_if_needed()
        prediction_result = predict_image(model, image_path)
        logger.info(f"Prediction result: {prediction_result}")

        if "error" in prediction_result:
            return jsonify(prediction_result), 400

        return jsonify(prediction_result), 200
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred while processing the image: {str(e)}"}), 500
    finally:
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                logger.info("Temporary image file removed")
            except Exception as e:
                logger.error(f"Error removing temporary file {image_path}: {e}")

# No development server in production
# Development server is for local testing only
if __name__ == '__main__':
    logger.warning("Running in development mode. Use gunicorn for production.")
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Running development server on host 0.0.0.0 and port {port}")
    app.run(host='0.0.0.0', port=port)
