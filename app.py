from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer, Conv2D
from tensorflow.keras.initializers import GlorotUniform, Zeros
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ================== MODEL COMPATIBILITY FIXES ==================
class FixedInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return super().from_config(config)

class FixedConv2D(Conv2D):
    @classmethod
    def from_config(cls, config):
        # Convert legacy dtype policy format
        if 'dtype' in config and isinstance(config['dtype'], dict):
            dtype_config = config['dtype']
            if dtype_config.get('class_name') == 'DTypePolicy':
                config['dtype'] = dtype_config['config']['name']
        return super().from_config(config)

# Custom objects mapping for legacy model support
CUSTOM_OBJECTS = {
    'InputLayer': FixedInputLayer,
    'Conv2D': FixedConv2D,
    'GlorotUniform': GlorotUniform,
    'Zeros': Zeros,
    'DTypePolicy': tf.keras.mixed_precision.Policy
}

# ================== MODEL PATHS AND LOADING ==================
MODEL_DIR = "/app/models"
MODEL_NAME = "autism_detection_model_(9.4).h5"
model_path = os.path.join(MODEL_DIR, MODEL_NAME)

# Debug: Log model path
print(f"Model path: {os.path.abspath(model_path)}")

try:
    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model at {model_path}: {e}")

# ================== YOLO INITIALIZATION ==================
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

# ================== HELPER FUNCTIONS ==================
def is_image_blurry(image, threshold=100):
    """Check if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def contains_human(image):
    """Detect if image contains a human using YOLO."""
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 is "person"
                return True
    return False

def preprocess_image(image_path, target_size=(299, 299)):
    """Preprocess image for model prediction."""
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

def predict_image(model, image_path):
    """Make prediction using the loaded model."""
    preprocess_result = preprocess_image(image_path)
    if "error" in preprocess_result:
        return preprocess_result

    image = preprocess_result["image"]
    prediction = model.predict(image)[0]
    predicted_class = np.argmax(prediction)
    class_labels = ['Autism', 'Normal']
    return {"prediction": class_labels[predicted_class]}

# ================== FLASK ROUTES ==================
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
