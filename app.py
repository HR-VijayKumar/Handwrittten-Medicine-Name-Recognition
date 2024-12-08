# app.py

import os
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load_model('prescription_classification_model.keras')

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the Excel file with medicine information
df = pd.read_excel('medicine_information.xlsx')

# Function to preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, color_mode='grayscale', target_size=(64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        # Ensure the upload directory exists
        upload_dir = 'static/uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # Save the file to the server
        image_path = os.path.join(upload_dir, file.filename)
        file.save(image_path)

        # Predict the medicine
        predicted_medicine_name = predict_medicine(image_path)

        # Get the medicine information from Excel
        medicine_info = get_medicine_info(predicted_medicine_name)

        return render_template('index.html', predicted_medicine_name=predicted_medicine_name, image_path=image_path, medicine_info=medicine_info)

    return render_template('index.html', predicted_medicine_name=None)

# Function to predict the medicine name
def predict_medicine(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_medicine_name = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_medicine_name

# Function to get medicine information
def get_medicine_info(medicine_name):
    info = df[df['Medicine Name'].str.lower() == medicine_name.lower()]
    if not info.empty:
        return info.to_dict(orient='records')[0]
    return {}

if __name__ == '__main__':
    app.run(debug=True)
