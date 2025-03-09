from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# Load models once at startup
MODEL_PATHS = {
    "resnet": r"D:\skin_disease_detection\trained_models\resnet101.h5",
    "densenet": r"D:\skin_disease_detection\trained_models\densenet121.h5",
    "efficientnet": r"D:\skin_disease_detection\trained_models\efficientnetb3.h5"
}
models = {name: tf.keras.models.load_model(path) for name, path in MODEL_PATHS.items()}

# Mapping from class names to indices (as used during training)
class_indices = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}
# Reverse mapping: index -> class label
idx2class = {v: k for k, v in class_indices.items()}

# Mapping for user-friendly display: label, full name, and description
USER_FRIENDLY_MAPPING = {
    "akiec": {
        "name": "Actinic Keratoses",
        "description": (
            "Actinic keratoses are rough, scaly patches on the skin caused by years of sun exposure. "
            "They can sometimes develop into skin cancer and should be monitored by a dermatologist."
        )
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": (
            "Basal cell carcinoma is the most common type of skin cancer. It is usually slow-growing "
            "and rarely metastasizes, but professional evaluation is recommended."
        )
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": (
            "Benign keratoses are non-cancerous skin growths. They are typically harmless, though any changes "
            "should be evaluated by a healthcare provider."
        )
    },
    "df": {
        "name": "Dermatofibroma",
        "description": (
            "Dermatofibromas are benign skin nodules that generally do not require treatment unless they "
            "cause discomfort or cosmetic concerns."
        )
    },
    "mel": {
        "name": "Melanoma",
        "description": (
            "Melanoma is a serious form of skin cancer that can be life-threatening if not detected early. "
            "Immediate consultation with a dermatologist is crucial."
        )
    },
    "nv": {
        "name": "Melanocytic Nevus",
        "description": (
            "Melanocytic nevi (moles) are usually benign. However, any noticeable changes in size, shape, "
            "or color should be examined by a professional."
        )
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": (
            "Vascular lesions are abnormalities of the blood vessels. While often benign, they may require "
            "treatment if symptomatic or for cosmetic reasons."
        )
    }
}

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image from disk, converts it from BGR to RGB,
    resizes it to target_size, preprocesses it using EfficientNet's preprocess_input,
    and expands dimensions to create a batch of size 1.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be read: " + image_path)
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image
    image = cv2.resize(image, target_size)
    # Preprocess using EfficientNet's preprocessing function
    image = preprocess_input(image)
    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)
    return image

def setup_prediction_routes(app: Flask):

    @app.route('/predict', methods=['POST'])
    def prediction():
        print("Request Headers:", request.headers)

        if 'image' not in request.files:
            print("No image part in the request.")
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        print("File Name:", image_file.filename)
        print("Content Type:", image_file.content_type)

        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_data = image_file.read()
        print("File Data (first 100 bytes):", file_data[:100])

        temp_dir = 'uploads'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(temp_dir, filename)

        with open(file_path, 'wb') as f:
            f.write(file_data)

        try:
            preprocessed_image = load_and_preprocess_image(file_path, target_size=(224, 224))
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        predictions = {}
        for name, model in models.items():
            pred = model.predict(preprocessed_image)[0]
            predictions[name] = pred
            print(f"Prediction from {name}: {pred}")

        # Ensemble predictions by averaging the probabilities
        avg_prediction = np.mean(list(predictions.values()), axis=0)
        predicted_index = np.argmax(avg_prediction)
        predicted_label = idx2class.get(predicted_index, str(predicted_index))
        friendly_info = USER_FRIENDLY_MAPPING.get(predicted_label, {})

        # Instead of ensemble confidence, display the maximum confidence value for the predicted class among the models
        max_confidence = max(prediction[predicted_index] for prediction in predictions.values())
        # Convert to percentage and format as a string (e.g. "99.99%")
        confidence_percent = f"{int(round(max_confidence * 100))}"


        return jsonify({
            "message": "Image received and processed successfully",
            "filename": filename,
            "predicted_disease": predicted_label,
            "confidence": confidence_percent,
            "disease_details": friendly_info
        })

setup_prediction_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
