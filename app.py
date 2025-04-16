from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
import cv2
import os
import shutil

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Wide CORS support

app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model
model = load_model('plant_disease_mobilenetv2.h5', compile=False)
# D:\full\backend\plant_disease_mobilenetv2.h5
# # Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# def preprocess_test_img(img_path):
#     test_img = cv2.imread(img_path)
#     if test_img is None:
#         print("Failed to load image at:", img_path)
#         return None

#     gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
#     resized_img = cv2.resize(gray_img, (128, 128))
#     img_array = np.array(resized_img) / 255.0
#     img_array = img_array.reshape(1, 128, 128, 1)
    
#     return img_array
# Preprocess the image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def clear_upload_folder():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img_array = preprocess_image(filepath)
    if img_array is None:
        return jsonify({'error': 'Invalid image'}), 400

    prediction = model.predict(img_array)
    class_index = int(np.argmax(prediction))  # Ensure it's a regular int
    print("Predicted class index:", class_index)
      # delete uploads folder
    clear_upload_folder()
    classs =  [
       "Bacterial Spot on Bell Pepper",
         "Healthy Bell Pepper Leaf",
        "Potato Early Blight",
     "Healthy Potato Leaf",
    "Potato Late Blight",
     "Tomato Bacterial Spot",
    "Tomato Early Blight",
   "Healthy Tomato Leaf",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites (Two-Spotted)",
    "Tomato Target Spot",
    "Tomato Mosaic Virus"
     ];
    return jsonify({'prediction': classs[int(class_index)]})

@app.route('/get_cure', methods=['POST'])
def get_cure():
    disease = request.json.get('disease')
    if not disease:
        return jsonify({'error': 'No disease name provided'}), 400

    # A dictionary containing cure and reasons for diseases
    diseases_info = {
        "Bacterial Spot on Bell Pepper": {
            "cure": "Apply copper-based fungicides or other antibacterial agents.",
            "reason": "Bacterial spot is caused by Xanthomonas bacteria. It spreads in wet and humid conditions."
        },
        "Healthy Bell Pepper Leaf": {
            "cure": "No treatment necessary, the plant is healthy.",
            "reason": "This disease-free pepper plant is in good condition."
        },
        "Potato Early Blight": {
            "cure": "Use fungicides containing chlorothalonil or mancozeb.",
            "reason": "Early blight is caused by the fungus Alternaria solani. It affects leaves and stems."
        },
        "Healthy Potato Leaf": {
            "cure": "No treatment necessary, the plant is healthy.",
            "reason": "This potato plant is free of diseases."
        },
        "Potato Late Blight": {
            "cure": "Fungicides with mefenoxam or chlorothalonil can help manage late blight.",
            "reason": "Late blight is caused by the pathogen Phytophthora infestans. It affects leaves and tubers."
        },
        "Tomato Bacterial Spot": {
            "cure": "Use copper-based fungicides and maintain proper plant spacing for airflow.",
            "reason": "Bacterial spot is caused by Xanthomonas bacteria. It thrives in wet conditions."
        },
        "Tomato Early Blight": {
            "cure": "Apply fungicides like chlorothalonil or mancozeb to control the blight.",
            "reason": "Early blight is caused by the fungus Alternaria solani, which attacks the leaves."
        },
        "Healthy Tomato Leaf": {
            "cure": "No treatment necessary, the plant is healthy.",
            "reason": "The tomato plant is disease-free and thriving."
        },
        "Tomato Late Blight": {
            "cure": "Apply fungicides containing mefenoxam or copper-based treatments.",
            "reason": "Late blight is caused by Phytophthora infestans and affects tomato leaves and fruit."
        },
        "Tomato Leaf Mold": {
            "cure": "Ensure proper ventilation and apply fungicides to manage the disease.",
            "reason": "Leaf mold is caused by the fungus Passalora fulva and thrives in high humidity."
        },
        "Tomato Septoria Leaf Spot": {
            "cure": "Apply fungicides like chlorothalonil to control leaf spot.",
            "reason": "Septoria leaf spot is caused by the fungus Septoria lycopersici and damages tomato leaves."
        },
        "Tomato Spider Mites (Two-Spotted)": {
            "cure": "Use insecticidal soap or miticides to control spider mites.",
            "reason": "Spider mites feed on tomato leaves and cause discoloration and spots."
        },
        "Tomato Target Spot": {
            "cure": "Use fungicides containing copper or chlorothalonil.",
            "reason": "Target spot is caused by the fungus Corynespora cassiicola and results in leaf lesions."
        },
        "Tomato Mosaic Virus": {
            "cure": "No cure available. Remove infected plants to prevent spread.",
            "reason": "Tomato mosaic virus is a viral infection that affects the growth and appearance of tomatoes."
        }
    }

    disease_info = diseases_info.get(disease, {"cure": "No cure information available for this disease.", "reason": "No reason information available for this disease."})
    return jsonify(disease_info)

if __name__ == '__main__':
    app.run(debug=True)
