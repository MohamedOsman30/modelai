from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer  # Import InputLayer for custom fix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# SOLUTION: Custom InputLayer class to handle legacy 'batch_shape' parameter
class FixedInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        # Convert legacy 'batch_shape' to 'batch_input_shape'
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return super().from_config(config)

# Define paths
MODEL_DIR = "/app/models"
MODEL_NAME = "autism_detection_model_(9.4).h5"
model_path = os.path.join(MODEL_DIR, MODEL_NAME)

# Debug: Log model path
print(f"Model path: {os.path.abspath(model_path)}")

# Load the autism detection model with custom fix
try:
    # Use custom InputLayer to handle legacy format
    model = load_model(
        model_path,
        custom_objects={'InputLayer': FixedInputLayer}
    )
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model at {model_path}: {e}")

# Rest of your code remains unchanged...
# [Keep your YOLO initialization, helper functions, and routes the same]
# ======================================================================
# Load YOLO model using OpenCV
YOLO_DIR = "/app/yolo"
weights_path = os.path.join(YOLO_DIR, "yolov3.weights")
cfg_path = os.path.join(YOLO_DIR, "yolov3.cfg")
names_path = os.path.join(YOLO_DIR, "coco.names")

# Load YOLO model
net = cv2.dnn.readNet(weights_path, cfg_path)

# Load COCO class labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to check if the image is blurry
def is_image_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian < threshold

# Function to check if the image contains a human using YOLO
def contains_human(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    human_detected = False
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 is "person"
                human_detected = True
                break
    print(f"Human detected: {human_detected}")
    return human_detected

# Preprocess the image
def preprocess_image(image_path, target_size=(299, 299)):
    original_image = cv2.imread(image_path)
    if original_image is None:
        return {"error": "Invalid image path or image could not be read."}

    if not contains_human(original_image):
        return {"error": "The picture is not suitable because it is not a human."}

    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return {"image": image}

# Prediction function
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

@app.route('/')
def home():
    return 'Autism Detection API is live.'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file temporarily
    temp_dir = "/app/temp"
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, "temp_image.jpg")
    file.save(image_path)
    print(f"Image saved temporarily at: {image_path}")

    try:
        prediction_result = predict_image(model, image_path)
        print(f"Prediction result: {prediction_result}")

        if "error" in prediction_result:
            return jsonify(prediction_result), 400

        return jsonify(prediction_result), 200
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred while processing the image."}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
            print("Temporary image file removed.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
