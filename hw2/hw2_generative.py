# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:24:21 2019

@author: kevin
"""
import csv
import math
import numpy as np
import pandas as pd
import sys

dim = 106
def load_data():
    #讀檔如果像這樣把路徑寫死交到github上去會馬上死去喔
    #還不知道怎寫請參考上面的連結
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

#    index = []
    for i in range(106):
         index.append(i)
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor
def train(x_train, y_train):
    cnt1 = 0
    cnt2 = 0
    
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    
    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
    return mu1, mu2, share_sigma, cnt1, cnt2

def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(share_sigma)

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred

x_train,y_train,x_test = load_data()
mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)
y = predict(x_train, mu1, mu2, shared_sigma, N1, N2)
y = np.around(y)
result = (y_train == y)
print('Train acc = %f' % (float(result.sum()) / result.shape[0]))
#predict x_test
prediction = predict(x_test, mu1, mu2, shared_sigma, N1, N2)
prediction = np.around(prediction)
#write csv
with open(sys.argv[6],'w',newline='') as csvfile:
     writer = csv.writer(csvfile)
     writer.writerow(['id','label'])
     for i in range(prediction.shape[0]):
          writer.writerow(["%d"%(i+1),"%d"%prediction[i]])
