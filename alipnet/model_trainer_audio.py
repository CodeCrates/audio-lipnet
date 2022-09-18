from sklearn import svm
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder

import seaborn
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt 

from DEFAULTS import AUDIO_FEATURES_CSV_FILE_CORRECTED,CLASSES
import print_statements as cprint


def create_deep_model(layers,activation_fn,x_train):
    model = Sequential()
    for (i,nodes) in enumerate(layers):
        if(i==0):
            model.add(Dense(nodes,activation=activation_fn,input_shape = (x_train.shape[1],)))
       
        else:
            model.add(Dense(nodes,activation=activation_fn))
    model.add(Dense(7,activation='softmax'))
    model.compile(optimizer='Adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
    return model
    

if __name__ == '__main__'           :

    data  = pd.read_csv(AUDIO_FEATURES_CSV_FILE_CORRECTED,index_col=False)
    data = data.drop(data.columns[0],axis=1)
    x_train,x_test, y_train,y_test = train_test_split(data.iloc[:,:-1],data['label'],test_size=0.2,random_state=0)

    print(x_train.shape)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    y_validate = y_test
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    model = create_deep_model([128,64,32],'relu',x_train)
    mcp_save = ModelCheckpoint('model_trained_with_audio.hdf5',save_best_only = True, monitor = 'val_loss', mode='min')
    model.fit(x_train,y_train,batch_size=2,epochs=50,validation_data=(x_test,y_test),callbacks=[mcp_save])

    predictions = model.predict(x_test)
    predictions = [CLASSES[list(each_row).index(max(each_row))] for each_row in predictions]
    predictions  = [CLASSES.index(each) for each in predictions]
    

    cprint.print_Debug_Statements("Accuracy",key = accuracy_score(y_validate,predictions))

    plt.Figure()
    seaborn.heatmap(confusion_matrix(y_validate,predictions),xticklabels=CLASSES,yticklabels=CLASSES,annot=True)
    plt.show()

