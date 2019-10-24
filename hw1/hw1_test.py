# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:26:53 2019

@author: kevin
"""



import math
import numpy as np
import pandas as pd
import sys
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

bias = 0.426948
w = np.load("hw1_weight.npy")

#result output
fout = open(sys.argv[2], 'w')
test = pd.read_csv(sys.argv[1])
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
