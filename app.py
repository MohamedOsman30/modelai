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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Log startup details
port = os.getenv('PORT', '5000')
logger.info(f"Starting Flask app on host 0.0.0.0 and port {port}")

# Lazy-load the model
model = None
model_lock = threading.Lock()

def load_model_in_background():
    global model
    with model_lock:
        if model is None:
            model_path = 'autism_detection_model_(9.4).h5'
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
            logger.info("Loading model in background...")
            model = load_model(model_path)
            logger.info("Model loaded successfully")

# Start model loading in a separate thread
threading.Thread(target=load_model_in_background, daemon=True).start()

def load_model_if_needed():
    with model_lock:
        if model is None:
            logger.info("Waiting for model to load...")
            load_model_in_background()
        return model

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

def is_image_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian < threshold

def preprocess_image(image_path, target_size=(299, 299)):
    original_image = cv2.imread(image_path)
    if original_image is None:
        return {"error": "Invalid image path or image could not be read."}

    if is_image_blurry(original_image):
        return {"error": "The image is too blurry."}

    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return {"image": image}

def predict_image(model, image_path):
    preprocess_result = preprocess_image(image_path)
    if "error" in preprocess_result:
        return preprocess_result

    image = preprocess_result["image"]
    prediction = model.predict(image)[0]
    predicted_class = np.argmax(prediction)
    class_labels = ['Autism', 'Normal']
    predicted_label = class_labels[predicted_class]
    return {"prediction": predicted_label}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = 'temp_image.jpg'
    file.save(image_path)
    logger.info(f"Image saved temporarily at: {image_path}")

    try:
        model = load_model_if_needed()
        prediction_result = predict_image(model, image_path)
        logger.info(f"Prediction result: {prediction_result}")

        if "error" in prediction_result:
            return jsonify(prediction_result), 400

        return jsonify(prediction_result), 200
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred while processing the image."}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.info("Temporary image file removed.")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Running development server on host 0.0.0.0 and port {port}")
    app.run(host='0.0.0.0', port=port)
