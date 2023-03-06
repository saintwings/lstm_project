import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from numpy import mean
from numpy import std
from numpy import dstack
from numpy import concatenate
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical





import time



# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None)
    #print(filepath)
    #print("row : ",len(dataframe))
    #print("column : ",len(dataframe.values[0]))
    #rint(dataframe.values)
    #exit()
    #print(type(dataframe.values))
    #print(dataframe.values.shape)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    #print(type(loaded))
    #print(loaded.shape)
    #print(loaded)

    return loaded

# load a dataset group, such as train or test
def load_dataset_group(prefix,window,stride):
    filepath = prefix
    suffix = '_w_' + str(window) + '_s_' + str(stride) + '.csv'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['_X_1' + suffix, '_X_2' + suffix]

    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(filepath + '_y'  + suffix)
    #print("X : ", X.shape)
    #print("y : ", y.shape)
    
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    window = 64#128
    stride = 32#32
    # load all train
    train_data = "B11"
    print("Train data: ",train_data)
    trainX_11, trainy_11 = load_dataset_group(prefix + 'data/Train/' + train_data,window,stride)
    print("trainX: ", trainX_11.shape, " | trainy: ", trainy_11.shape)


    train_data = "B12"
    print("Train data: ",)
    trainX_12, trainy_12 = load_dataset_group(prefix + 'data/Train/' + train_data,window,stride)
    print("trainX: ", trainX_12.shape, " | trainy: ", trainy_12.shape)

    trainX = concatenate((trainX_11,trainX_12),axis=0)
    trainy = concatenate((trainy_11,trainy_12),axis=0)

    #print("trainX : ", trainX.shape)
    #print("trainy : ", trainy.shape)


    # load all test
    test_data = "B17"
    print("Test data: ",test_data)
    testX, testy = load_dataset_group(prefix + 'data/Test/' + test_data,window,stride)
    testy = testy.astype(int)
    print("testX: ", testX.shape, " | testy: ", testy.shape)

    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    #print("trainy",trainy)
    #print("testy",testy)



    return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    # define model
    verbose, epochs, batch_size = 1, 3, 128
    n_features, n_outputs = trainX.shape[2], trainy.shape[1]
    # reshape data into time steps of sub-sequences
    
    n_steps, n_length = 1, 64
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    
    #print('trainX.shape',trainX.shape)
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
    input_shape=(None,n_length,n_features)))
    #model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    #model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    # fit network
    print("Training...")
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    print("Testing...")
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
    return accuracy





def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def run_experiment(repeats=1):
    start_time = time.time()
    # load data
    trainX, trainy, testX, testy = load_dataset()
        # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

    print("Overall process time : ",(time.time() - start_time))


if __name__ == "__main__":
    run_experiment()