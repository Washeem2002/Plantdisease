from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
import cv2
import os
import shutil
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = 'uploads'

model = load_model('plant_disease_mobilenetv2.h5', compile=False)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    class_index = int(np.argmax(prediction))
    clear_upload_folder()

    class_labels = [
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
    ]

    return jsonify({'prediction': class_labels[class_index]})

@app.route('/get_cure', methods=['POST'])
def get_cure():
    disease = request.json.get('disease').get('prediction')
    if not disease:
        return jsonify({'error': 'No disease name provided'}), 400

    diseases_info = {
    "Bacterial Spot on Bell Pepper": {
        "reason": "Bacterial spot is caused by Xanthomonas bacteria. It spreads in wet and humid conditions.",
        "treatment": {
            "steps": [
                "Remove and destroy infected leaves.",
                "Avoid overhead watering to reduce leaf wetness.",
                "Apply copper-based fungicides or antibacterial agents every 7–10 days.",
                "Improve air circulation by spacing plants properly."
            ]
        },
        "prevention": {
            "steps": [
                "Use certified disease-free seeds.",
                "Avoid working with wet plants.",
                "Rotate crops every season.",
                "Sanitize garden tools regularly."
            ]
        },
        "Image":["static/Images/Bacterial_Spot_on_Bell_Pepper_1.jpg","static/Images/Bacterial_Spot_on_Bell_Pepper_2.jpg","static/Images/Bacterial_Spot_on_Bell_Pepper_3.jpg","static/Images/Bacterial_Spot_on_Bell_Pepper_4.jpg"]
    },
    "Healthy Bell Pepper Leaf": {
        "reason": "This disease-free pepper plant is in good condition.",
        "treatment": {
            "steps": ["No treatment necessary, the plant is healthy."]
        },
        "prevention": {
            "steps": [
                "Maintain healthy soil and proper watering practices.",
                "Monitor for early signs of disease regularly."
            ]
        },
        "Image":["static/Images/Healthy_Bell_Pepper_Leaf_1.jpg","static/Images/Healthy_Bell_Pepper_Leaf_2.jpg","static/Images/Healthy_Bell_Pepper_Leaf_3.jpg","static/Images/Healthy_Bell_Pepper_Leaf_4.jpg"]
    },
    "Potato Early Blight": {
        "reason": "Early blight is caused by the fungus Alternaria solani. It affects leaves and stems.",
        "treatment": {
            "steps": [
                "Remove affected leaves to reduce spore spread.",
                "Rotate crops to prevent fungal buildup.",
                "Apply fungicides containing chlorothalonil or mancozeb every 7 days.",
                "Avoid overhead irrigation and water early in the day."
            ]
        },
        "prevention": {
            "steps": [
                "Use resistant potato varieties.",
                "Plant in well-drained soil and full sunlight.",
                "Avoid excessive nitrogen fertilizers.",
                "Ensure good air circulation between plants."
            ]
        },
        "Image":["static/Images/Potato_Early_Blight_1.jpg","static/Images/Potato_Early_Blight_2.jpg","static/Images/Potato_Early_Blight_3.jpg","static/Images/Potato_Early_Blight_4.jpg"]
    },
    "Healthy Potato Leaf": {
        "reason": "This potato plant is free of diseases.",
        "treatment": {
            "steps": ["No treatment necessary, the plant is healthy."]
        },
        "prevention": {
            "steps": [
                "Monitor plants weekly.",
                "Use clean tools and healthy soil."
            ]
        },
        "Image":["static/Images/Healthy_Potato_Leaf_1.jpg","static/Images/Healthy_Potato_Leaf_2.jpg","static/Images/Healthy_Potato_Leaf_3.jpg","static/Images/Healthy_Potato_Leaf_4.jpg"]
    },
    "Potato Late Blight": {
        "reason": "Late blight is caused by the pathogen Phytophthora infestans. It affects leaves and tubers.",
        "treatment": {
            "steps": [
                "Remove and destroy infected plant parts immediately.",
                "Avoid excess moisture; water plants early in the day.",
                "Apply fungicides with mefenoxam or chlorothalonil every 5–7 days.",
                "Ensure proper spacing and airflow around plants."
            ]
        },
        "prevention": {
            "steps": [
                "Use certified seed potatoes.",
                "Avoid planting in areas with previous outbreaks.",
                "Do not leave potato debris in the field.",
                "Monitor for early signs after rainfall."
            ]
        },
        "Image":["static/Images/Potato_Late_Blight_1.jpg","static/Images/Potato_Late_Blight_2.jpg","static/Images/Potato_Late_Blight_3.jpg","static/Images/Potato_Late_Blight_4.jpg"]
    },
    "Tomato Bacterial Spot": {
        "reason": "Bacterial spot is caused by Xanthomonas bacteria. It thrives in wet conditions.",
        "treatment": {
            "steps": [
                "Remove and dispose of infected leaves.",
                "Avoid overhead watering and reduce humidity.",
                "Apply copper-based fungicides every 7–10 days.",
                "Space plants to improve air circulation."
            ]
        },
        "prevention": {
            "steps": [
                "Avoid working with plants when wet.",
                "Use pathogen-free seeds and transplants.",
                "Sterilize tools regularly."
            ]
        },
        "Image":["static/Images/Tomato_Bacterial_Spot_1.jpg","static/Images/Tomato_Bacterial_Spot_2.jpg","static/Images/Tomato_Bacterial_Spot_3.jpg","static/Images/Tomato_Bacterial_Spot_4.jpg"]
    },
    "Tomato Early Blight": {
        "reason": "Early blight is caused by the fungus Alternaria solani, which attacks the leaves.",
        "treatment": {
            "steps": [
                "Remove diseased leaves and improve air circulation.",
                "Rotate tomato crops yearly to prevent recurrence.",
                "Spray fungicides like chlorothalonil or mancozeb every 7–10 days.",
                "Avoid watering leaves directly."
            ]
        },
        "prevention": {
            "steps": [
                "Mulch plants to prevent soil splash.",
                "Stake or cage tomatoes to keep leaves dry.",
                "Avoid excessive watering."
            ]
        },
        "Image":["static/Images/Tomato_Early_Blight_1.jpg","static/Images/Tomato_Early_Blight_2.jpg","static/Images/Tomato_Early_Blight_3.jpg","static/Images/Tomato_Early_Blight_4.jpg"]
    },
    "Healthy Tomato Leaf": {
        "reason": "The tomato plant is disease-free and thriving.",
        "treatment": {
            "steps": ["No treatment necessary, the plant is healthy."]
        },
        "prevention": {
            "steps": [
                "Inspect regularly and prune when needed.",
                "Maintain proper nutrition and watering."
            ]
        },
        "Image":["static/Images/Tomato_Early_Blight_1.jpg","static/Images/Tomato_Early_Blight_2.jpg","static/Images/Tomato_Early_Blight_3.jpg","static/Images/Tomato_Early_Blight_4.jpg"]
    },
    "Tomato Late Blight": {
        "reason": "Late blight is caused by Phytophthora infestans and affects tomato leaves and fruit.",
        "treatment": {
            "steps": [
                "Remove and destroy infected plant parts.",
                "Avoid excessive humidity and water early in the day.",
                "Apply fungicides containing mefenoxam or copper-based treatments every 5–7 days.",
                "Do not compost infected plants."
            ]
        },
        "prevention": {
            "steps": [
                "Use resistant tomato varieties.",
                "Avoid planting near infected potatoes.",
                "Keep garden clean of plant debris."
            ]
        },
        "Image":["static/Images/Tomato_Early_Blight_1.jpg","static/Images/Tomato_Early_Blight_2.jpg","static/Images/Tomato_Early_Blight_3.jpg","static/Images/Tomato_Early_Blight_4.jpg"]
    },
    "Tomato Leaf Mold": {
        "reason": "Leaf mold is caused by the fungus Passalora fulva and thrives in high humidity.",
        "treatment": {
            "steps": [
                "Ensure good ventilation and reduce indoor humidity.",
                "Remove and destroy infected leaves.",
                "Apply fungicides according to label directions."
            ]
        },
        "prevention": {
            "steps": [
                "Avoid overcrowding of plants.",
                "Water at the base of plants to keep foliage dry."
            ]
        },"Image":["static/Images/Tomato_Early_Blight_1.jpg","static/Images/Tomato_Early_Blight_2.jpg","static/Images/Tomato_Early_Blight_3.jpg","static/Images/Tomato_Early_Blight_4.jpg"]
    },
    "Tomato Septoria Leaf Spot": {
        "reason": "Septoria leaf spot is caused by the fungus Septoria lycopersici and damages tomato leaves.",
        "treatment": {
            "steps": [
                "Remove and discard affected leaves.",
                "Avoid overhead watering.",
                "Spray fungicides like chlorothalonil every 7 days.",
                "Practice crop rotation and good garden hygiene."
            ]
        },
        "prevention": {
            "steps": [
                "Avoid wetting leaves during irrigation.",
                "Space plants for airflow.",
                "Clean up debris at the end of the season."
            ]
        },
        "Image":["static/Images/Tomato_Early_Blight_1.jpg","static/Images/Tomato_Early_Blight_2.jpg","static/Images/Tomato_Early_Blight_3.jpg","static/Images/Tomato_Early_Blight_4.jpg"]
    },
    "Tomato Spider Mites (Two-Spotted)": {
        "reason": "Spider mites feed on tomato leaves and cause discoloration and spots.",
        "treatment": {
            "steps": [
                "Spray plants with water to dislodge mites.",
                "Use insecticidal soap or miticides.",
                "Apply neem oil weekly until infestation subsides."
            ]
        },
        "prevention": {
            "steps": [
                "Keep plants well-watered to reduce stress.",
                "Regularly inspect undersides of leaves.",
                "Encourage natural predators like ladybugs."
            ]
        },
        "Image":["static/Images/Tomato_Early_Blight_1.jpg","static/Images/Tomato_Early_Blight_2.jpg","static/Images/Tomato_Early_Blight_3.jpg","static/Images/Tomato_Early_Blight_4.jpg"]
    },
    "Tomato Target Spot": {
        "reason": "Target spot is caused by the fungus Corynespora cassiicola and results in leaf lesions.",
        "treatment": {
            "steps": [
                "Remove affected leaves and debris from around the plant.",
                "Avoid wetting foliage during irrigation.",
                "Apply copper or chlorothalonil-based fungicides regularly."
            ]
        },
        "prevention": {
            "steps": [
                "Ensure proper spacing and air circulation.",
                "Remove weeds and maintain field hygiene."
            ]
        },
        "Image":["static/Images/Tomato_Early_Blight_1.jpg","static/Images/Tomato_Early_Blight_2.jpg","static/Images/Tomato_Early_Blight_3.jpg","static/Images/Tomato_Early_Blight_4.jpg"]
    },
    "Tomato Mosaic Virus": {
        "reason": "Tomato mosaic virus is a viral infection that affects the growth and appearance of tomatoes.",
        "treatment": {
            "steps": [
                "Remove and destroy infected plants immediately.",
                "Disinfect tools after handling plants.",
                "Avoid handling plants when they are wet.",
                "Use virus-resistant tomato varieties in the future."
            ]
        },
        "prevention": {
            "steps": [
                "Wash hands and tools before working with plants.",
                "Use certified virus-free seeds.",
                "Control weeds and avoid tobacco use near plants."
            ]
        },
        "Image":["static/Images/Tomato_Early_Blight_1.jpg","static/Images/Tomato_Early_Blight_2.jpg","static/Images/Tomato_Early_Blight_3.jpg","static/Images/Tomato_Early_Blight_4.jpg"]
    }
}



    disease_info = diseases_info.get(disease, {"cure": "No cure information available.", "reason": "No reason available.", "products": [], "similar": []})
    return jsonify(disease_info)


  

if __name__ == '__main__':
    app.run(debug=True)