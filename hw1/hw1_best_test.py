# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:48:53 2019

@author: kevin
"""
import tensorflow as tf
import math
import numpy as np
import pandas as pd
from keras.models import Model
import sys


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


model = tf.contrib.keras.models.load_model('hw1_best.h5')



x_test = convert2testset(sys.argv[1])
result = model.predict(x_test)
    
fout = open(sys.argv[2], 'w')
test = pd.read_csv(sys.argv[1])
id_name = []                  
for name in test['id']:
    if (name not in id_name):
        id_name.append(name)
print('id,value', file = fout) 
    
for i in range(len(id_name)):
    print('%s,%f' %(id_name[i], result[i]), file = fout)
fout.close()
    
