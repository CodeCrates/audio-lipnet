#python 3.6

import numpy as np
import librosa
import os
from moviepy import editor

def convert_vedio_to_audio(video):
    try:
        video = editor.VideoFileClip(video)
        audio = video.audio
        audio.write_audiofile(AUDIO_FILE_NAME)
        return True,AUDIO_FILE_NAME
    except Exception as e:
        cprint.print_Error_statements("conversion from video to audio failed",key = e)
        return False,AUDIO_FILE_NAME
    

def extract_feature(audio_file_name):
    X, sample_rate = librosa.load(audio_file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
        
    return mfccs,chroma,mel,contrast,tonnetz
  
def process_audio_features(mfccs,chroma,mel,contrast,tonnetz):   

        features_array = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        return features_array
    
def get_audio_feature_vector(video_location):
    isConverted,audio_file_name = convert_vedio_to_audio(video_location)
    if(isConverted):
         mfccs,chroma,mel,contrast,tonnetz =  extract_feature(audio_file_name)
         feature_vector = process_audio_features(mfccs,chroma,mel,contrast,tonnetz)
         feature_vector  = np.around(feature_vector,decimals=3)
         return feature_vector
    else:
        raise ValueError('Video conversion failed...')
