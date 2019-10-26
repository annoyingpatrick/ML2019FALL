# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:16:20 2019

@author: kevin
"""
import numpy as np
import pandas as pd
import sys
import csv

def load_data():
  
    x_train = pd.read_csv(sys.argv[3])
    x_test = pd.read_csv(sys.argv[5])

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv(sys.argv[4], header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)
    return x_train, y_train, x_test

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)

def normalize(x_train, x_test):
    
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0,1,2,3,5]
#    for i in range(106):
#         index.append(i)
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor

#load weight&bias
npz_file=np.load("logistic_weight.npz")
w=npz_file["w"]
b=npz_file["b"]
#load data
x_train, y_train, x_test = load_data()
x_train, x_test = normalize(x_train, x_test)

#predict
pred_test=sigmoid(np.dot(x_test,w)+b)
pred_test=np.around(pred_test)
"""
#output result
with open('output.csv','w',newline='') as csvfile:
     writer = csv.writer(csvfile)
     writer.writerow(['id','label'])
     for i in range(pred_test.shape[0]):
          writer.writerow(["%d"%(i+1),"%d"%pred_test[i]])
"""

fout = open(sys.argv[6], 'w')
print("id,label",file=fout)
for i in range(len(pred_test)):
     print("%d,%d"%(i+1,pred_test[i]),file=fout)
fout.close()