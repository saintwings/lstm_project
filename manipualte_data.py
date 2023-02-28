import csv
from pandas import read_csv
import numpy as np

def rewrite_data_file():
    with open('DataSetForPython/B1XTrain.csv') as csv_file:
        #csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.reader(csv_file, quoting = csv.QUOTE_NONNUMERIC,
                            delimiter = ',')
        line_count = 0

        data_1 = []
        data_2 = []
        for row in csv_reader:
            line_count+=1
            data_count = 1
            data_1_column = []
            data_2_column = []
            for data in row:
                if(data_count % 2) == 1 :
                    data_1_column.append(data)
                else:
                    data_2_column.append(data)

                data_count += 1
            
            data_1.append(data_1_column)
            data_2.append(data_2_column)

    with open('DataSetForPython/B1_X1_Train.csv', 'w', newline= '') as file:
        writer = csv.writer(file, quoting = csv.QUOTE_NONNUMERIC)
        writer.writerows(data_1)

    with open('DataSetForPython/B1_X2_Train.csv', 'w', newline= '') as file:
        writer = csv.writer(file, quoting = csv.QUOTE_NONNUMERIC)
        writer.writerows(data_2)

def read_data(path,filepath_X_1,filepath_X_2, filepath_y):
    dataframe_X_1 = read_csv(path + filepath_X_1, header=None)
    dataframe_X_2 = read_csv(path + filepath_X_2, header=None)
    dataframe_y = read_csv(path + filepath_y, header=None)
    return dataframe_X_1.values,dataframe_X_2.values, dataframe_y.values

def split_sequence(data_X_1,data_X_2, data_y,windowLength, stride):
    X_1,X_2, y = list(), list(),list()

    if(len(data_X_1) != len(data_X_2) != len(data_y)):
        print("len(data_X_1) != len(data_X_2) != len(data_y)")
        exit()
    for i in range(len(data_X_1)):
        #print("i = ",i)
        zero_value_index_X_1 = np.where(data_X_1[i] == 0)
        zero_value_index_X_2 = np.where(data_X_2[i] == 0)
        zero_value_index_y = np.where(data_y[i] == 0)
        data_X_1_new = np.delete(data_X_1[i], zero_value_index_X_1[0])
        data_X_2_new = np.delete(data_X_2[i], zero_value_index_X_2[0])
        data_y_new = np.delete(data_y[i], zero_value_index_y[0])

        last_index = len(data_X_1_new) 
        start_index = 0
        while((start_index + windowLength) <= last_index):
            #print(start_index,start_index + windowLength,last_index)
            X_1_s = data_X_1_new[start_index : start_index + windowLength]
            X_2_s = data_X_2_new[start_index : start_index + windowLength]
            y_s = data_y_new[start_index + windowLength -1]
            start_index += stride
            #print("i",i,"X_1",X_1_s,"X_2",X_2_s,"y",y_s)

            X_1.append(X_1_s)
            X_2.append(X_2_s)
            y.append(y_s)

        
        # if i == 0:
        #     exit()
    print(len(X_1))
    print(len(X_2))
    print(len(y))

    # print(y)

    # exit()

    with open('X_1_w_'+ str(windowLength) +'_s_'+ str(stride) +'.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(X_1)
    
    with open('X_2_w_'+ str(windowLength) +'_s_'+ str(stride) +'.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(X_2)
    
    with open('y_w_'+ str(windowLength) +'_s_'+ str(stride) +'.csv', 'w') as f:
        write = csv.writer(f,delimiter="\n")
        write.writerow(y)
        





    #for i in range


if __name__ == "__main__":
    path = 'DataSetForPython/train/'
    X_1, X_2, y = read_data(path ,'B1_X1_train.csv','B1_X2_train.csv','B1_Y_train.csv')
    #print(X)
    #print(X.shape)
    #print(y)
    #print(y.shape)

    split_sequence(X_1,X_2,y,10,1)
