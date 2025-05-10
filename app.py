from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import random
from models import sampling

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORT_FOLDER'] = 'static/reports'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


# Load the trained global model
global_model = tf.keras.models.load_model('global_model.h5', custom_objects={'sampling': sampling})


# Function to preprocess image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize based on model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Define reference links
reference_links = [
    {"title": "🧪 What is Liver Fibrosis?", "url": "https://www.merckmanuals.com/home/liver-and-gallbladder-disorders/fibrosis-and-cirrhosis-of-the-liver/fibrosis-of-the-liver"},
    {"title": "🧪 Treatment & Management:", "url": "https://www.fda.gov/news-events/press-announcements/fda-approves-first-treatment-patients-liver-scarring-due-fatty-liver-disease"},
    {"title": "📑 Research Papers", "url": "https://pubmed.ncbi.nlm.nih.gov/28051792/"},
    {"title": "👨🏻‍🏫 Patient Stories", "url": "https://britishlivertrust.org.uk/information-and-support/support-for-you/your-stories/mikes-story-i-was-seeing-colours-differently"},
    {"title": "📊 Latest News on Liver Health", "url": "https://www.verywellhealth.com/how-a-liver-elastography-test-works-8736281"}
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contactus.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('prediction.html', error="No file part")

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('prediction.html', error="Invalid file type")

    try:
        # Read image in-memory
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        # Preprocess image
        img_array = preprocess_image(image)
        # Predict using the VAE-CNN model
        _, y_pred_probs = global_model.predict(img_array)
        fibrosis_stage = f"F{np.argmax(y_pred_probs)}"
        # Select 3 random reference links
        selected_links = random.sample(reference_links, 3)
        return render_template('prediction.html', fibrosis_stage=fibrosis_stage, reference_links=selected_links)
    except Exception as e:
        return render_template('prediction.html', error=str(e))

@app.route('/download')
def download_file():
    file_path = r"static\understanding-your-fibroscan-results.pdf"  # Path to your file
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    
    app.run(debug=True)
    
