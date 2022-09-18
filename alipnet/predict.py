# python 3.6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix

CLASSES = ['about', 'bottle', 'dog', 'english', 'good', 'people', 'today']

##AUDIO_VEDIO_COMBINED_FEATURES_CSV_FILE = r'Datasets/audio_video_features_7x15_videos.csv'

audio_model_weights = 'model_trained_with_audio_features.hdf5'


def create_model(input_shape, list_target_classes, layers, activation_fn):
    model = Sequential()
    model.add(
        Dense(units=layers[0], activation=activation_fn, input_shape=input_shape))
    model.add(Dense(layers[1], activation=activation_fn))
    model.add(Dense(layers[2], activation=activation_fn))
    model.add(Dense(7, activation='softmax'))
    return model


def get_model(audio_or_video, input_shape, weights_file_name):

    model = create_model(input_shape, CLASSES, [128, 64, 32], 'relu')
    model.load_weights(weights_file_name)
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_model():
    audio_model = get_model('audio', input_shape=(
        193,), weights_file_name=audio_model_weights)

    return audio_model


def get_predictions(model, features):
    print(features.shape)
    prediction_vector = model.predict([[features]])
    print("prediction vector " + str(prediction_vector))
    #prediction = [list(each).index(each.max()) for each in prediction]
    prediction = list(prediction_vector[0]).index(prediction_vector[0].max())
    return prediction, prediction_vector


def get_results(audio_features):
    audio_model = load_model()
    audio_prediction = get_predictions(audio_model, audio_features)
    print(audio_prediction)
    return CLASSES[audio_prediction]
