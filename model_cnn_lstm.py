from numpy import mean
from numpy import std
from numpy import dstack
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
    print(type(loaded))
    print(loaded.shape)
    #print(loaded)

    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['B1_X1_'+group+'.csv', 'B1_X2_'+group+'.csv']

    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/B1_Y_'+group+'.csv')
    print("X : ", X.shape)
    print("y : ", y.shape)
    exit()
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'DataSetForPython/')
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'DataSetPython/')
    # zero-offset class values
    # trainy = trainy - 1
    # testy = testy - 1
    # # one hot encode y
    # trainy = to_categorical(trainy)
    # testy = to_categorical(testy)
    # return trainX, trainy, testX, testy
    return 0,0,0,0

def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()


if __name__ == "__main__":
    run_experiment()