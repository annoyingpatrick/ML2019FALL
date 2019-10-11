# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 23:33:44 2019

@author: kevin
"""
import tensorflow as tf
import math
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras import losses
from keras.layers import Dense, Input, Add
from keras.models import Model


def convert2testset(file_name):
     a = pd.read_csv(file_name)
     a.fillna(value=0 , inplace=True)
     a = np.array(a)
     c=[]
     for i in range(np.shape(a)[0]):
          if(a[i][1]=="RAINFALL"):
               pass
          b=[]
          for j in range(2,11):
               if(a[i][j]==0):
                    b.append(0.0)
               elif (a[i][j] == "NR"):
                    b.append(0.0)
               elif (a[i][j][-1:]=="x" or a[i][j][-1:]=="#" or a[i][j][-1:]=="*" ):
                    b.append(float(a[i][j][:-1]))
               else:
                    b.append(float(a[i][j]))
          c.append(b)
          del(b)
     d=[]
     for i in range(np.shape(c)[0]):
          if(i%18==0):
               a=[]
          a.extend(c[i])
          if(i%18==17):
               d.append(a)
               del(a)
     return np.array(d)


def readdata(data):
    
	# 把有些數字後面的奇怪符號刪除
	for col in list(data.columns[2:]):
		data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
	data = data.values
	
	# 刪除欄位名稱及日期
	data = np.delete(data, [0,1], 1)
	
	# 特殊值補0
	data[ data == 'NR'] = 0
	data[ data == ''] = 0
	data[ data == 'nan'] = 0
	data = data.astype(np.float)

	return data

def extract(data):
	N = data.shape[0] // 18

	temp = data[:18, :]
    
    # Shape 會變成 (x, 18) x = 取多少hours
	for i in range(1, N):
		temp = np.hstack((temp, data[i*18: i*18+18, :]))
	return temp

def valid(x, y):
	if y <= 2 or y > 100:
		return False
	for i in range(9):
		if x[9,i] <= 2 or x[9,i] > 100:
			return False
	return True

def parse2train(data):
	x = []
	y = []
	
	# 用前面9筆資料預測下一筆PM2.5 所以需要-9
	total_length = data.shape[1] - 9
	for i in range(total_length):
		x_tmp = data[:,i:i+9]
		y_tmp = data[9,i+9]
		if valid(x_tmp, y_tmp):
			x.append(x_tmp.reshape(-1,))
			y.append(y_tmp)
	# x 會是一個(n, 18, 9)的陣列， y 則是(n, 1) 
	x = np.array(x)
	y = np.array(y)
	return x,y
def MSE(y_test,pred):
    n = np.size(y_test)
    sum = 0.0 
    for i in range(n):
        sum += (y_test[i]-pred[i])**2
    error = tf.sqrt(sum)
    error /= n
    return error

def convert2testset(file_name):
     a = pd.read_csv(file_name)
     a.fillna(value=0 , inplace=True)
     a = np.array(a)
     c=[]
     for i in range(np.shape(a)[0]):
          if(a[i][1]=="RAINFALL"):
               pass
          b=[]
          for j in range(2,11):
               if(a[i][j]==0):
                    b.append(0.0)
               elif (a[i][j] == "NR"):
                    b.append(0.0)
               elif (a[i][j][-1:]=="x" or a[i][j][-1:]=="#" or a[i][j][-1:]=="*" ):
                    b.append(float(a[i][j][:-1]))
               else:
                    b.append(float(a[i][j]))
          c.append(b)
          del(b)
     d=[]
     for i in range(np.shape(c)[0]):
          if(i%18==0):
               a=[]
          a.extend(c[i])
          if(i%18==17):
               d.append(a)
               del(a)
     return np.array(d)



    
    
year1_pd = pd.read_csv(r'C:\Users\kevin\Downloads\ml2019fall-hw1\year1-data.csv')
year1 = readdata(year1_pd)
train_data1 = extract(year1)
year2_pd = pd.read_csv(r'C:\Users\kevin\Downloads\ml2019fall-hw1\year2-data.csv')
year2 = readdata(year2_pd)
train_data2 = extract(year2)
train_data = np.hstack((train_data1,train_data2))
train_x, train_y = parse2train(train_data)
    
inputs = Input(shape=(162,)) 
dense_1 = Dense(20, activation='relu')(inputs)
outputs = Dense(1, activation='relu')(dense_1)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=0.0005),metrics=[MSE])
model.summary()
    
train_history=model.fit(train_x, train_y,validation_split=0.25,epochs=100 , batch_size=50,shuffle = True, initial_epoch=0, verbose=1)
    
scores=model.evaluate(train_x,train_y)  
x_test = convert2testset(r'C:\Users\kevin\Downloads\ml2019fall-hw1\testing_data.csv')
result = model.predict(x_test)
    
fout = open('result.csv', 'w')
test = pd.read_csv(r'C:\Users\kevin\Downloads\ml2019fall-hw1\testing_data.csv')
id_name = []                  
for name in test['id']:
    if (name not in id_name):
        id_name.append(name)
print('id,value', file = fout) 
    
for i in range(len(id_name)):
    print('%s,%f' %(id_name[i], result[i]), file = fout)
fout.close()
    