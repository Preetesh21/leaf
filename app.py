from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

#Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

INIT_LR = 1e-3
default_image_size = tuple((128, 128))
image_size = 0
width=128
height=128
depth=3

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def model_predict(img_path, model):
    img = convert_image_to_array(img_path) #target_size must agree with what the trained model expects!!
    np_image_list = np.array(img, dtype=np.float16) / 225.0
    # Preprocessing the image
    image=img.reshape(1,128,128,3)
    preds = model.predict(image)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned
        print(preds)
        disease=['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight' ,'Potato___Late_blight', 'Potato___healthy','Tomato_Bacterial_spot', 'Tomato_Early_blight' ,'Tomato_Late_blight','Tomato_Leaf_Mold' ,'Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus','Tomato_healthy']
        maxi=np.argmax(preds)
        print(maxi)
        return disease[maxi]
    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
