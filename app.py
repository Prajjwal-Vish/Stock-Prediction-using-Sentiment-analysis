from flask import Flask, request, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from io import BytesIO
from prepare_data import prepare_data
from get_metrics import get_metrics
import base64

app = Flask(__name__)

app.secret_key = 'my_secret_key'

# Paths
MODEL_PATH = r"C:\Users\prajjwal\Desktop\College-project\data\model\model1.h5"  # Path to the saved model
UPLOAD_FOLDER = r"C:\Users\prajjwal\Desktop\College-project\uploaded_file"      # Folder to save uploaded files

# Set up the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_model_function():
    return keras.models.load_model(MODEL_PATH)

def plot_graph(y_true, y_pred):
    plt.figure(figsize=(10, 6), facecolor='green')
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red')
    plt.xlabel('Samples')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction: True vs Predicted')
    plt.legend()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

ALLOWED_EXTENSIONS = {'csv'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for handling the file upload and predictions

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML form

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the file is present in the request
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    
    df = pd.read_csv(file)
    x, y = prepare_data(df)
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))

    model = load_model_function()
    y_pred = model.predict(x)

    metrics = get_metrics(y,y_pred)
    # Plot the results and encode the image
    img_base64 = plot_graph(y, y_pred)

    # Render the results
    return render_template('results.html', img_data=img_base64, metrics = metrics)

if __name__ == '__main__':
    app.run(debug=False)
