import csv
from pandas import read_csv
import numpy as np

def load_data_from_raw(prefix,path_X, path_y):
    
    dataframe_X = read_csv(prefix + path_X, header=None)



    dataframe_y = read_csv(prefix + path_y, header=None)

    # print(len(dataframe_X.values[0]))
    # print(len(dataframe_X.values[1]))
    print(len(dataframe_y.values[0]))

    split_sequence_2(dataframe_X.values[0],dataframe_X.values[1],dataframe_y.values[0],100,25)
    

def split_sequence_2(data_X_1,data_X_2, data_y,windowLength, stride):
    X_1,X_2, y = list(), list(),list()
    if(len(data_X_1) != len(data_X_2) != len(data_y)):
        print("len(data_X_1) != len(data_X_2) != len(data_y)")
        exit()

    last_index = len(data_X_1) 
    start_index = 0

    while((start_index + windowLength) <= last_index):
            #print(start_index,start_index + windowLength,last_index)
        X_1_s = data_X_1[start_index : start_index + windowLength]
        X_2_s = data_X_2[start_index : start_index + windowLength]
        y_s = data_y[start_index + windowLength -1]
        start_index += stride
        #print("i",i,"X_1",X_1_s,"X_2",X_2_s,"y",y_s)

        X_1.append(X_1_s)
        X_2.append(X_2_s)
        y.append(y_s)
    
    with open(prefix + 'X_1_w_'+ str(windowLength) +'_s_'+ str(stride) +'.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(X_1)
    
    with open(prefix + 'X_2_w_'+ str(windowLength) +'_s_'+ str(stride) +'.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(X_2)
    
    with open(prefix + 'y_w_'+ str(windowLength) +'_s_'+ str(stride) +'.csv', 'w') as f:
        write = csv.writer(f,delimiter="\n")
        write.writerow(y)


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
        






def rewrite_data_file(path,output_path_1,output_path_2):
    with open(path) as csv_file:
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

    with open(output_path_1, 'w', newline= '') as file:
        writer = csv.writer(file, quoting = csv.QUOTE_NONNUMERIC)
        writer.writerows(data_1)

    with open(output_path_2, 'w', newline= '') as file:
        writer = csv.writer(file, quoting = csv.QUOTE_NONNUMERIC)
        writer.writerows(data_2)

def rewrite_data_file_1(path,output_path):
    with open(path) as csv_file:
        #csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.reader(csv_file, quoting = csv.QUOTE_NONNUMERIC,
                            delimiter = ',')
        line_count = 0

        data_1 = list()
        data_2 = list()
        for row in csv_reader:
            line_count+=1
            data_count = 1
            data_1_column = list()
            data_2_column = list()
            for data in row:
                if(data_count % 2) == 1 :
                    data_1_column.append(data)
                else:
                    data_2_column.append(data)

                data_count += 1
            



    data_all = np.array([data_1_column,data_2_column])

    print("shape",data_all.shape)
        


    with open(output_path, 'w') as file:
    
        writer = csv.writer(file)

        writer.writerows(data_all)



def read_data(path,filepath_X_1,filepath_X_2, filepath_y):
    dataframe_X_1 = read_csv(path + filepath_X_1, header=None)
    dataframe_X_2 = read_csv(path + filepath_X_2, header=None)
    dataframe_y = read_csv(path + filepath_y, header=None)
    return dataframe_X_1.values,dataframe_X_2.values, dataframe_y.values



    #for i in range


def check_data():
    path = "data/Train/X_1_w_100_s_25.csv"
    data_X_1 = read_csv(path, header=None)
    print("data_X_1", data_X_1.values.shape)

    path = "data/Train/X_2_w_100_s_25.csv"
    data_X_2 = read_csv(path, header=None)
    print("data_X_2", data_X_2.values.shape)

    path = "data/Train/y_w_100_s_25.csv"
    data_y = read_csv(path, header=None)
    print("data_y", data_y.values.shape)

    ##################

    path = "data/Test/X_1_w_100_s_25.csv"
    data_X_1 = read_csv(path, header=None)
    print("data_X_1", data_X_1.values.shape)

    path = "data/Test/X_2_w_100_s_25.csv"
    data_X_2 = read_csv(path, header=None)
    print("data_X_2", data_X_2.values.shape)

    path = "data/Test/y_w_100_s_25.csv"
    data_y = read_csv(path, header=None)
    print("data_y", data_y.values.shape)

def replace_value(find, replace):
    path = "data/Test/B13YTest.csv"
    with open(path) as csv_file:
        #csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.reader(csv_file, quoting = csv.QUOTE_NONNUMERIC
                            )
        
        for row in csv_reader:
            data = np.array(row)
            
    #print(data)

    indexs = np.where(data == 2)
    #print(indexs[0])

    for index in indexs:
        data[index] = 0
    
    print(data)
    print(data.shape)



    output_path  = "data/Test/B13YTest_1.csv"
    with open(output_path, 'w') as file:
    
        writer = csv.writer(file)

        writer.writerow(data)

        
    

        

def rewrite_file_1():
    ## 1
    path = 'DataSetForPython/B13XTest.csv'
    output_path = 'data/Test/B13XTest.csv'
    rewrite_data_file_1(path,output_path)


if __name__ == "__main__":
    ## 0
    #prefix = "data/"
    #load_data_from_raw(prefix, 'B11XTrain_Full.csv','B11YTrain_Full.csv')

    prefix = "data/Test/"
    load_data_from_raw(prefix, 'B13XTest.csv','B13YTest.csv')

    #check_data()
    #replace_value(2,0)



    ## 2
    # path = 'DataSetForPython/train/'
    # X_1, X_2, y = read_data(path ,'B1_X1_train.csv','B1_X2_train.csv','B1_Y_train.csv')


    # split_sequence(X_1,X_2,y,10,1)
