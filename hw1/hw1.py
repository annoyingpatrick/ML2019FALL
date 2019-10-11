# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 00:02:46 2019

@author: kevin
"""
import math
import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense
import matplotlib.pyplot as plt
def convert_trainset(file_name):
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

def build_testset(data):
	x = []
	y = []
	
	# 用前面9筆資料預測下一筆PM2.5 所以需要-9
	total_length = 500
	for i in range(total_length):
		x_tmp = data[:,i*9:(i+1)*9]
		if (1):
			x.append(x_tmp.reshape(-1,))
	# x 會是一個(n, 18, 9)的陣列
	x = np.array(x)
	return x

def MSE(y_test,prediction):#mean square error
     length = np.size(y_test)
     total = 0.0
     for i in range(length):
          total = total + abs(prediction[i]-y_test[i])**2
     error = np.sqrt(total)/length;
     return error

def minibatch(x, y):
    # 打亂data順序
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    
    # 訓練參數以及初始化
    batch_size = 64
    lr = 1e-3
    lam = 0.001
    beta_1 = np.full(x[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(x[0].shape, 0.99).reshape(-1, 1)
    w = np.full(x[0].shape, 0.1).reshape(-1, 1)
    bias = 0.1
    m_t = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8
    
    for num in range(1000):
        for b in range(int(x.shape[0]/batch_size)):
            t+=1
            x_batch = x[b*batch_size:(b+1)*batch_size]
            y_batch = y[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            loss = y_batch - np.dot(x_batch,w) - bias
            
            # 計算gradient
            g_t = np.dot(x_batch.transpose(),loss) * (-2) +  2 * lam * np.sum(w)
            g_t_b = loss.sum(axis=0) * (2)
            m_t = beta_1*m_t + (1-beta_1)*g_t 
            v_t = beta_2*v_t + (1-beta_2)*np.multiply(g_t, g_t)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            w_0 = np.copy(w)
            
            # 更新weight, bias
            w -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
            bias -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)
            

    return w, bias

year1_pd = pd.read_csv(r'C:\Users\kevin\Downloads\ml2019fall-hw1\year1-data.csv')

year1 = readdata(year1_pd)
train_data = extract(year1)
x_train, y_train = parse2train(train_data)
w, bias = minibatch(x_train, y_train)


year2_pd = pd.read_csv(r'C:\Users\kevin\Downloads\ml2019fall-hw1\year2-data.csv')

year2 = readdata(year1_pd)
val_data = extract(year1)
x_val, y_val = parse2train(val_data)

result = np.dot(x_val,w)
result = result + bias
val_mse = MSE(y_val,result)
print("val_mse = %f" %val_mse)

#result output
fout = open('result.csv', 'w')
test = pd.read_csv(r'C:\Users\kevin\Downloads\ml2019fall-hw1\testing_data.csv')
test_ = readdata(test)
test_data = extract(test_)
testset = build_testset(test_data)
result_2 = np.dot(testset,w)+bias
id_name = []                  
for name in test['id']:
    if (name not in id_name):
        id_name.append(name)
print('id,value', file = fout) 

for i in range(len(id_name)):
    print('%s,%f' %(id_name[i], result_2[i]), file = fout)
fout.close()
