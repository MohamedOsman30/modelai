from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the autism detection model
model_path = 'autism_detection_model_(9.4).h5'
model = load_model(model_path)

# Function to check if the image is blurry
def is_image_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian < threshold

# Preprocessing the image
def preprocess_image(image_path, target_size=(299, 299)):
    # Load the original image using OpenCV
    original_image = cv2.imread(image_path)
    if original_image is None:
        return {"error": "Invalid image path or image could not be read."}

    # Optional: Check if the image is blurry (you can remove this if not needed)
    if is_image_blurry(original_image):
        return {"error": "The image is too blurry."}

    # Load and preprocess the image for the model
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)  # Normalizing the input
    return {"image": image}

# Prediction function
def predict_image(model, image_path):
    preprocess_result = preprocess_image(image_path)
    if "error" in preprocess_result:
        return preprocess_result  # Return the error message

    image = preprocess_result["image"]
    prediction = model.predict(image)[0]
    predicted_class = np.argmax(prediction)
    class_labels = ['Autism', 'Normal']
    predicted_label = class_labels[predicted_class]

    return {"prediction": predicted_label}

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    image_path = 'temp_image.jpg'
    file.save(image_path)
    print(f"Image saved temporarily at: {image_path}")

    try:
        # Make prediction
        prediction_result = predict_image(model, image_path)
        print(f"Prediction result: {prediction_result}")

        if "error" in prediction_result:
            return jsonify(prediction_result), 400

        return jsonify(prediction_result), 200
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred while processing the image."}), 500
    finally:
        # Remove the temporary file
        if os.path.exists(image_path):
            os.remove(image_path)
            print("Temporary image file removed.")


