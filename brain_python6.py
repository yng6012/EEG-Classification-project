#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf
import os
import sys
import random
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softsign
from tensorflow.keras.layers import LSTM, Dense

class brain_python:
    def __init__(self, n_times, n_frequency, n_windows, n_outputs):
        self.line_of_data = n_times*n_frequency
        self.line_of_data_times = n_times
        self.line_of_frequency = n_frequency
        self.window = n_windows
        self.outputs = n_outputs
        
    def EEG_model(self,input_shape,drop_out=0):
        self.model = Sequential([
        LSTM(units=40 ,dropout=drop_out),
        LSTM(units=40 ,dropout=drop_out),
        LSTM(units=40 ,dropout=drop_out),
        Dense(self.outputs,activation="softmax")
        ])
        self.model.compile(loss="MSE",optimizer = Adam(lr=0.001), metrics=['accuracy'])
        
        self.model.build(input_shape)
        self.model.summary()
        
    def Train_model(self,x_train, y_train,x_test,y_test, batch_size):
        train_history=self.model.fit(x_train, y_train,batch_size=batch_size)
        test_history=self.model.evaluate(x_test, y_test, verbose=2)
        return train_history, test_history
        
    def label_info(self,path):
        label_list = os.listdir(path)
        label_dic = {w:i for i,w in enumerate(label_list)}
        label_onehot = {i:w for i,w in enumerate(label_list)}
        
        return label_list, label_dic, label_onehot
        
    def preprocessing(self,path,equal_count_data = True):
        number_of_data = 0
        label_list = os.listdir(path)
        label_dic = {w:i for i,w in enumerate(label_list)}
        
        if equal_count_data == True:
        
            for label_list_name in label_list:
                if number_of_data<=len(os.listdir(os.path.join(path,label_list_name))):
                    number_of_data = len(os.listdir(os.path.join(path,label_list_name)))

            most_number_of_data = number_of_data
            number_of_data *= len(label_list)

            data_array = np.zeros((number_of_data, self.line_of_data_times, self.line_of_frequency))
            data_label = np.zeros((number_of_data, len(label_list)))
            data_number = np.zeros(number_of_data,dtype='U32')
            
        elif equal_count_data == False:
            for label_list_name in label_list:
                number_of_data += len(os.listdir(os.path.join(path,label_list_name)))

            data_array = np.zeros((number_of_data, self.line_of_data_times, self.line_of_frequency))
            data_label = np.zeros((number_of_data, len(label_list)))
            data_number = np.zeros(number_of_data,dtype='U32')
            
        else:
            print("error equal_count_data, please retry True or False.")
        
        data_num = 0
        reference_num = 1
        
        for label_list_name in label_list:
            print(label_list_name)
            for number_name in os.listdir(os.path.join(path,label_list_name)) :
                with open(os.path.join(path,label_list_name,number_name).replace("\\","/"),'r') as data:
                    lines = data.read().splitlines()
                    lines = lines[0:self.line_of_data]
                    lines = list(map(float,lines))
                    
                data_array[data_num,:] = np.reshape(lines,(self.line_of_data_times,self.line_of_frequency))
                data_label[data_num,label_dic[label_list_name]] = 1.0
                data_number[data_num] = number_name

                data_num += 1
                
            if equal_count_data == True:
                while data_num<(most_number_of_data*reference_num):
                    random_number = random.randint(most_number_of_data*(reference_num-1),data_num-1)
                    data_array[data_num,:] = data_array[random_number]
                    data_label[data_num,label_dic[label_list_name]] = 1.0
                    data_number[data_num] = data_number[random_number]
                    data_num += 1
                reference_num+=1
                
                    
        return data_array, data_label, data_number
    
    def data_multiple(self,data_array,data_array_label, label_list, important_index,normal_range_index,step):
        
        control_loop =  (np.max(important_index)-self.window+1)
        discrimination_loop = (np.max(important_index)-np.min(important_index))<=self.window

        start_loop = control_loop*((np.max(important_index)-self.window)>=0)
        number_loop = self.window - (np.max(important_index)-np.min(important_index))+control_loop*((np.max(important_index)-self.window)<=0)

        new_array = np.array([],dtype=np.uint32)
        new_normal_array = np.array([],dtype=np.uint32)

        if discrimination_loop:
            for i in range(0,number_loop,step):
                new_array=np.append(new_array,np.arange(self.window)+start_loop+i)
            new_array = new_array.reshape(-1,self.window)

            test_step = int((normal_range_index[1]-normal_range_index[0]-self.window)/new_array.shape[0])

            if test_step == 0:
                print("error : The range of 'ragne_index' is too small. Please expand the range.")
                exit()
            else :
                for i in range(normal_range_index[0],normal_range_index[0]+test_step*new_array.shape[0],test_step):
                    new_normal_array = np.append(new_normal_array,np.arange(self.window)+i)

                new_normal_array = new_normal_array.reshape(-1,self.window)


        else :
            print("error : important_index range over than window.")
            exit()

        new_data = np.zeros([new_array.shape[0]*data_array.shape[0],self.window,self.line_of_frequency])
        new_data_label = np.zeros([new_array.shape[0]*data_array_label.shape[0],data_array_label.shape[1]])

        for i in range(data_array.shape[0]):
            if label_list[np.argmax(data_array_label[i])]=='normal':
                start_index = i*new_normal_array.shape[0]
                end_index = (i+1)*new_normal_array.shape[0]
                new_data[start_index:end_index] = data_array[i,new_normal_array]
                new_data_label[start_index:end_index] = data_array_label[i]
            else:
                start_index = i*new_array.shape[0]
                end_index = (i+1)*new_array.shape[0]
                new_data[start_index:end_index] = data_array[i,new_array]
                new_data_label[start_index:end_index] = data_array_label[i]

        return new_data, new_data_label

