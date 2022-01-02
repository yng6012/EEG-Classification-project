#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf
import os
import sys
import random
from tensorflow.nn import rnn_cell

class brain_python:
    def __init__(self, n_times, n_frequency, n_windows, n_outputs, n_neurons, learning_rate):
        self.line_of_data = n_times*n_frequency
        self.line_of_data_times = n_times
        self.line_of_frequency = n_frequency
        self.window = n_windows
        self.outputs = n_outputs
        
        self.X = tf.placeholder(tf.float32,[None, self.line_of_frequency, self.window])
        self.Y = tf.placeholder(tf.float32,[None, self.outputs])

        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        #for loop의 layer를 [128,64,32]와 같이 바꿔 봄으로 서, num_units 만져보기.
        self.cells = [rnn_cell.LSTMCell(num_units = neurons, forget_bias = 1.0, activation=tf.nn.softsign) for neurons in n_neurons]
        self.cells_drop = [rnn_cell.DropoutWrapper(cell, input_keep_prob = self.keep_prob, output_keep_prob = self.keep_prob) for cell in self.cells]
        self.multi_layer_cell = rnn_cell.MultiRNNCell(self.cells_drop)
        self.lstm_outputs, states = tf.nn.dynamic_rnn(self.multi_layer_cell, self.X, dtype = tf.float32)

        self.stacked_lstm_outputs = tf.layers.dense(self.lstm_outputs[:,:,-1], self.outputs)
        self.softmax_outputs = tf.nn.softmax(self.stacked_lstm_outputs)

        self.predictions = tf.reshape(self.softmax_outputs,[-1,self.outputs])

        self.sce = tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=self.predictions)
        
        self.train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.sce)
    
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
    
    def train(self,sess,train_data,label_data, train_keep_prob):
        _,loss = sess.run([self.train_op,self.sce], feed_dict = {self.X: train_data, self.Y: label_data, self.keep_prob: train_keep_prob})
        return loss
    
    def accuracy(self,sess,test_data,test_label, test_keep_prob,label_onehot, test_label_onehot):
        
        correct_list = np.array([],dtype='i8') 
        fail_list = np.array([],dtype='i8')
        unknown_list = np.array([],dtype='i8')
        y_pred = sess.run(self.predictions, feed_dict = {self.X : test_data, self.keep_prob : test_keep_prob})
        accuracy_result = y_pred>0.9
        sum_accuracy = 0
        index = 0
        
        for i,j in zip(accuracy_result,test_label):
            if any(i) :
                if test_label_onehot[np.argmax(j)] in label_onehot[np.argmax(i)]:
                    sum_accuracy += 1
                    correct_list = np.append(correct_list,index)
                else :
                    fail_list = np.append(fail_list,index)
            else :
                unknown_list = np.append(unknown_list,index)
            index += 1
            
        accuracy = sum_accuracy/(test_label.shape[0])
    
        return accuracy, correct_list, fail_list, unknown_list, y_pred

#     def accuracy(self,sess,test_data,test_label, test_keep_prob,label_list, test_label_list):
        
#         correct_list = np.array([],dtype='i8') 
#         fail_list = np.array([],dtype='i8')
#         unknown_list = np.array([],dtype='i8')
#         y_pred = sess.run(self.predictions, feed_dict = {self.X : test_data, self.keep_prob : test_keep_prob})
#         new_y_pred = np.zeros([y_pred.shape[0],len(test_label_list)])
#         y_pred_index = 0
        
#         for k in y_pred:
#             accuracy_result = np.zeros(len(test_label_list))
#             for i in test_label_list:
#                 sum_percent = 0
#                 for j in label_list:
#                     if i in j:
#                         sum_percent += k[label_list.index(j)]
#                 accuracy_result[test_label_list.index(i)] = sum_percent
#             new_y_pred[y_pred_index] = accuracy_result
#             y_pred_index += 1
            
#         new_y_pred = new_y_pred>=0.9
            
#         sum_accuracy = 0
#         index = 0
        
#         for i in np.equal(new_y_pred,test_label):
#             if i.all() :
#                 sum_accuracy += 1
#                 correct_list = np.append(correct_list,index)
#             index += 1
            
#         accuracy = sum_accuracy/(test_label.shape[0])
    
#         return accuracy, correct_list,y_pred

