import csv
from pandas import read_csv
import numpy as np

def generate_data(prefix_path, group, prefix_name, path_X1, path_X2, path_y, window, stride):
    
    full_path_X1 = prefix_path + group + prefix_name + path_X1
    full_path_X2 = prefix_path + group + prefix_name + path_X2
    full_path_y = prefix_path + group + prefix_name + path_y

    print("Load : ", full_path_X1)
    print("Load : ", full_path_X2)
    print("Load : ", full_path_y)


    dataframe_X1 = read_csv(full_path_X1, header=None)
    dataframe_X2 = read_csv(full_path_X2, header=None)
    dataframe_y = read_csv(full_path_y, header=None)


    print("Spliting...")
    X_1, X_2, y = split_sequence(dataframe_X1.values, dataframe_X2.values, dataframe_y.values, window, stride)

    prefix_save_file = "Dataset_Window/"

    with open(prefix_save_file + group + prefix_name + '_X_1_w_'+ str(window) +'_s_'+ str(stride) +'.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(X_1)
    
    with open(prefix_save_file + group + prefix_name + '_X_2_w_'+ str(window) +'_s_'+ str(stride) +'.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(X_2)
    
    with open(prefix_save_file + group + prefix_name + '_y_w_'+ str(window) +'_s_'+ str(stride) +'.csv', 'w') as f:
        write = csv.writer(f,delimiter="\n")
        write.writerow(y)
    print("Saved +++")

def split_sequence(data_X_1, data_X_2, data_y, window, stride):
    X_1,X_2, y = list(), list(),list()
    if(len(data_X_1) != len(data_X_2) != len(data_y)):
        print("len(data_X_1) != len(data_X_2) != len(data_y)")
        exit()

    for i in range(len(data_X_1)):
        #print(i)
        last_index = len(data_X_1[i]) 
        start_index = 0
        #print("last index: ", last_index)
    
        while((start_index + window) <= last_index):
                #print(start_index,start_index + windowLength,last_index)
            X_1_s = data_X_1[i][start_index : start_index + window]
            X_2_s = data_X_2[i][start_index : start_index + window]
            y_s = data_y[i][start_index + window -1]
            start_index += stride
            #print("i",i,"X_1",X_1_s,"X_2",X_2_s,"y",y_s)

            X_1.append(X_1_s)
            X_2.append(X_2_s)
            y.append(y_s)
    
    return X_1,X_2,y
    


if __name__ == "__main__":

    window = 32#16#128
    stride = 8#4#32

    ### Train ##
    train_data = ["B11", "B12"]
    for train in train_data:
        generate_data("Dataset_Augmentation/", 
                      "Train_Set/", 
                      train,
                      "_XTrain_Feature1.csv",
                      "_XTrain_Feature2.csv",
                      "_YTrain.csv",
                      window,
                      stride)
    
    ### Test ###
    test_data = ["B13", "B14", "B15", "B16", "B17"]
    for test in test_data :
        generate_data("Dataset_Augmentation/", 
                    "Test_Set/", 
                    test,
                    "XTest_Mean_H.csv",
                    "XTest_Mean_V.csv",
                    "YTest_Mean.csv",
                    window,
                    stride)
    
    print("Finished ##############################")
    
    