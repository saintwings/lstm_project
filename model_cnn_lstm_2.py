import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from numpy import mean
from numpy import std
from numpy import dstack
from numpy import concatenate
from numpy import array
from numpy import argmax
from numpy import arange
from pandas import read_csv
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

import time

import matplotlib.pyplot as plt

def load_file(filepath):
    dataframe = read_csv(filepath, header=None)

    return dataframe.values

def load_data(files, group, window, stride):

    #print(len(files))
    X = array([]).reshape(0,window,2)
    y = array([]).reshape(0,1)
    for file in files:
        file_feature_X1_name = 'data/' + group + '/' + file + '_X_1_w_' + str(window) + '_s_' + str(stride) + '.csv'  
        file_feature_X2_name = 'data/' + group + '/' + file + '_X_2_w_' + str(window) + '_s_' + str(stride) + '.csv'  
        file_label_name = 'data/' + group + '/' + file + '_y_w_' + str(window) + '_s_' + str(stride) + '.csv'  
        
        X_loaded = list()

        X_loaded.append(load_file(file_feature_X1_name))
        X_loaded.append(load_file(file_feature_X2_name))
        X_loaded = dstack(X_loaded)
        X = concatenate((X, X_loaded),axis=0)

        y_loaded = load_file(file_label_name)

        y = concatenate((y, y_loaded))
        #print("X ",X_loaded.shape)
        #print("y ",y_loaded.shape)
    print(group)
    print("X ",X.shape)
    print("y ",y.shape)

    y_encoded = to_categorical(y)
    #print(y_encoded)
    print("y_encoded ",y_encoded.shape)
    print("111111111sss")
    return X, y_encoded, y

    #exit()
        
        # print(file_feature_X1_name)
        # print(file_feature_X2_name)
        # print(file_label_name)



def train_model(X_train, y_train, model_path, window,epochs):

    verbose, epochs, batch_size = 1, epochs, 128
    n_features, n_outputs = X_train.shape[2], y_train.shape[1]


    n_steps, n_length = 1, window
    X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))


    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=11, activation='relu', padding="same"),
    input_shape=(None,n_length,n_features)))
    model.add(TimeDistributed(Dropout(0.9)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, padding="same")))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    # fit network
    print("Training...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    model.save(model_path)


def evaluate_model(X_test, y_test, model_path,window):
    n_steps, n_length = 1, window
    batch_size = 128
    n_features, n_outputs = X_test.shape[2], y_test.shape[1]
    X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
    model = load_model(model_path)

    results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print("test loss, test acc:", results)

    predictions = model.predict(X_test)
    print("predictions shape:", predictions.shape)
    #print(predictions)

    #predictions[predictions <= 0.5] = 0.
    #predictions[predictions > 0.5] = 1.
    #print(predictions)

    class_labels = argmax(predictions, axis=1)
    print(class_labels)
    print(len(class_labels))
    # predictions = model.predict(X_test)

    return class_labels


def train(model_path, prefix_train_data, window, stride, epochs):
    
    X_train, y_train, _ = load_data(prefix_train_data , 'Train', window, stride)
    train_model(X_train, y_train, model_path, window,epochs)

def test(model_path,perfix_test_data,window, stride):
    X_test, y_test, y_law = load_data(perfix_test_data , 'Test', window, stride)

    
    y_predic = evaluate_model(X_test, y_test, model_path, window)


    ## plot ##
    y_law = y_law.reshape(1,-1)
    list_y_law = y_law[0].tolist()
    #s = arange(0,len(y_predic),1)

    plt.plot(y_predic, 'r')
    plt.plot(list_y_law, 'b')
    plt.show()


if __name__ == "__main__":
    window = 16
    stride = 4

    model_path = 'model/model_' + str(window) + "_" + str(stride)
    prefix_train_data = ['B11','B12']
    
    perfix_test_data = ['B15']

 

    ###############################

    mode = 2      # 1 = train , 2 = test

    if mode == 1 :
        train(model_path, prefix_train_data, window, stride, 10)
    elif mode == 2 :
        test(model_path,perfix_test_data,window, stride)

