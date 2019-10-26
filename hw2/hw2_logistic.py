# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:02:19 2019

@author: kevin
"""

import numpy as np
import pandas as pd

def load_data():
    #讀檔如果像這樣把路徑寫死交到github上去會馬上死去喔
    #還不知道怎寫請參考上面的連結
    x_train = pd.read_csv('X_train')
    x_test = pd.read_csv('X_test')

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv('Y_train', header = None)
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

def train(x_train, y_train):
    b = 0.0
    w = np.zeros(x_train.shape[1])
    lr = 0.5
    epoch = 500
    b_lr = 0
    w_lr = np.ones(x_train.shape[1])
    
    for e in range(epoch):
        z = np.dot(x_train, w) + b
        pred = sigmoid(z)
        loss = y_train - pred

        b_grad = -1*np.sum(loss)
        w_grad = -1*np.dot(loss, x_train)

        b_lr += b_grad**2
        w_lr += w_grad**2


        b = b-lr/np.sqrt(b_lr)*b_grad
        w = w-lr/np.sqrt(w_lr)*w_grad

        if(e+1)%50 == 0:
            loss = -1*np.mean(y_train*np.log(pred+1e-100) + (1-y_train)*np.log(1-pred+1e-100))
            print('epoch:{}\nloss:{}\n'.format(e+1,loss))
    return w, b

x_train, y_train, x_test = load_data()
x_train, x_test = normalize(x_train, x_test)
w, b = train(x_train, y_train)
#train accuracy
pred_train=sigmoid(np.dot(x_train,w)+b)
pred_train=np.around(pred_train)
result = (pred_train==y_train)
print('Train acc = %f' % (float(result.sum()) / result.shape[0]))
#predict x_test 
pred=sigmoid(np.dot(x_test,w)+b)
pred=np.around(pred)
#output pred

with open('output.csv','w',newline='') as csvfile:
     writer = csv.writer(csvfile)
     writer.writerow(['id','label'])
     for i in range(pred.shape[0]):
          writer.writerow(["%d"%(i+1),"%d"%pred[i]])
