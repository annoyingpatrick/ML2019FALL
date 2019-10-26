# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 22:45:48 2019

@author: kevin
"""

from sklearn.ensemble import GradientBoostingClassifier
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys

def load_data():

    x_train = pd.read_csv(sys.argv[3])
    x_test = pd.read_csv(sys.argv[5])

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv(sys.argv[4], header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

def normalize(x_train, x_test):
    
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [1]

    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor

x_train, y_train, x_test = load_data()
    
x_train, x_test = normalize(x_train, x_test)
#delete fnlwgt
#index=[]

#x_test=np.delete(x_test,index,axis=1)
#x_train=np.delete(x_train,index,axis=1)

#parse valid/train
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.2, random_state=15)

#instantiate classifier using default params
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
y=gbc.predict(x_train)
result = (y_train == y)
print('Train acc = %f' % (float(result.sum()) / result.shape[0]))
y1=gbc.predict(x_valid)
result = (y_valid == y1)
print('valid acc = %f' % (float(result.sum()) / result.shape[0]))
#predict test
prediction=gbc.predict(x_test)
#write csv
with open(sys.argv[6],'w',newline='') as csvfile:
     writer = csv.writer(csvfile)
     writer.writerow(['id','label'])
     for i in range(prediction.shape[0]):
          writer.writerow(["%d"%(i+1),"%d"%prediction[i]])


