import librosa
import numpy as np
import pandas as pd
import os
from moviepy import editor



import print_statements as cprint
from DEFAULTS import CLASSES,\
                AUDIO_FILE_NAME,\
                TRAIN_DATA_LOCATION,\
                AUDIO_FEATURES_CSV_FILE
"""
"""
# def print_shape(npArrayName,npArray):
#     cprint.print_Debug_Statements(npArrayName,key = npArray.shape)

"""
parameters : video : video location 
"""
def convert_vedio_to_audio(video):
    try:
        video = editor.VideoFileClip(video)
        audio = video.audio
        audio.write_audiofile(AUDIO_FILE_NAME)
        return True
    except Exception as e:
        cprint.print_Error_statements("conversion from video to audio failed",key = e)
        return False
    
"""
"""
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
 
 """
 parameters : audio features : mfccs,chroma,mel,contrast,tonnetz
 returns :  final feature vector of length 193
 """
def process_audio_features(mfccs,chroma,mel,contrast,tonnetz):   

        features_array = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        return features_array
# Length of the feature vector is 193
# mfccs : (40,)
# chroma : (12,)
# mel : (128,)
# contrast : (7,)
# tonnetz : (6,)
# features_array : (193,)
        # print_shape("mfccs",mfccs)
        # print_shape("chroma",chroma)
        # print_shape("mel",mel)
        # print_shape("contrast",contrast)
        # print_shape("tonnetz",tonnetz)
        # print_shape("features_array",features_array)

def process(TRAIN_DATA_LOCATION):
    
                temp_dictionary = {}
                features_Dataframe = pd.DataFrame()
                
                for each_word in os.listdir(TRAIN_DATA_LOCATION):
                        video_count = 0
                        
                        #print("#########" + each_word + "##########")
                        cprint.print_Heading(each_word)
                        
                        word_folder = os.path.join(TRAIN_DATA_LOCATION,each_word)

                        print("Current Directory  :" + word_folder )

                        for each_video in os.listdir(word_folder):
                                        
                                        video_count+=1
                                        
                                        # print("Processing video " + each_video)
                                        # print("Video count " + str(i))
                                        cprint.print_Debug_Statements("Processing videos... video count",key=video_count)
                                       
                                        if(convert_vedio_to_audio(os.path.join(word_folder,each_video))):
                                            mfccs,chroma,mel,contrast,tonnetz = extract_feature(AUDIO_FILE_NAME)
                                            feature_vector = process_audio_features(mfccs,chroma,mel,contrast,tonnetz)
                                         
                                            feature_vector  = np.around(feature_vector,decimals=3)
                                            
                                            temp_dictionary = dict(enumerate(feature_vector))
                                            temp_dictionary.update({'label':each_word})
                                            
                                            features_Dataframe = features_Dataframe.append(temp_dictionary, ignore_index=True)
                                        
                                        else:
                                             cprint.print_Seperator(style='?')
                                             cprint.print_Multiple_Debug_Statement("word","video_number",word=each_word,video_number = video_count) 
                                             cprint.print_Seperator(style='?')              
                        
                return features_Dataframe

            
if __name__ == "__main__":
        
                features_Dataframe = process(TRAIN_DATA_LOCATION)
                features_Dataframe = features_Dataframe.sample(frac=1).reset_index(drop=True)
                features_Dataframe.to_csv(AUDIO_FEATURES_CSV_FILE,index=False)
              
                print("Done.....")