import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

NUM_SAMPLES = 22050 # in 1 second (librosa default)
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

model = load_model("models/voice.h5")