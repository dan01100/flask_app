import base64
import io
from flask import Flask
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request, jsonify, render_template
from PIL import Image
import xgboost
from xgboost import Booster, XGBClassifier
import pandas as pd
import librosa
import os
app = Flask(__name__)


#Import models
import flower_classifier
import fight_predictor
import number_recognition

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/flower_classifier")
def flowers():
    return render_template('flower_classifier.html')

@app.route("/fight_predictor")
def fight():
    return render_template('fight_predictor.html')

@app.route("/number_recognition")
def voice():
    return render_template('number_recognition.html')



@app.route("/flower_classifier_predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    preprocessed_image = flower_classifier.preprocess_image(image, target_size=(224, 224))

    prediction = flower_classifier.model.predict(preprocessed_image).squeeze().tolist()

    response = {
        'prediction' : prediction
    }

    return jsonify(response)


@app.route("/fight_predictor_predict", methods=["POST"])
def predict2():
    message = request.get_json(force=True)
    winner = message["f1"]
    loser = message["f2"]

    input = fight_predictor.toVec(winner, loser, fight_predictor.fighters)

    if input == None:

        response = {
            'prediction' : [0,0]
        }

        return jsonify(response)

    input = input[0]

    input = np.array(input).reshape((1,-1))
    input = fight_predictor.std_scale.transform(input)


    prediction = fight_predictor.model.predict_proba(input).squeeze().tolist()

    response = {
        'prediction' : prediction
    }

    return jsonify(response)


@app.route("/number_recognition_predict", methods=["POST"])
def predict3():

    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)

    with open("audiofile.wav", "wb") as f:
        f.write(decoded)

    signal, sr = librosa.load("audiofile.wav")

    #Truncate signal
    signal = signal[:number_recognition.NUM_SAMPLES]

    
    MFFCs = librosa.feature.mfcc(signal, 
                                n_mfcc=number_recognition.N_MFCC, 
                                hop_length=number_recognition.HOP_LENGTH, 
                                n_fft=number_recognition.N_FFT)

    prediction = number_recognition.model.predict(MFFCs.T.reshape((1,44,13, 1))).squeeze().tolist()

    response = {
        'prediction' : prediction
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()

