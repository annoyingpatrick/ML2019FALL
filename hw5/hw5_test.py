import sys
import csv
import pandas as pd
import numpy as np
import re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import Recurrent, LSTM, SimpleRNN
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
	
testX = pd.read_csv(sys.argv[1])

x_test=[]

for i in range(testX.shape[0]):
  tmp= re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ", testX.iloc[i,1])
  x_test.append(tmp)

#tokenizer = Tokenizer(num_words = 5000)



x_test_seq = tokenizer.texts_to_sequences(x_test)


x_test = sequence.pad_sequences(x_test_seq,maxlen=50)

model = Sequential()
model.add(Embedding(output_dim = 512, input_dim = 5000, 
                    input_length = 50))
model.add(Dropout(0.5))
model.add(LSTM(512, activation = 'relu', return_sequences = True,
                       dropout = 0.2, recurrent_dropout = 0.2))
model.add(LSTM(256, activation = 'relu', return_sequences = True,
                       dropout = 0.2, recurrent_dropout = 0.2))
model.add(LSTM(256, activation = 'relu', dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()
model.compile(loss = 'binary_crossentropy', 
                  optimizer = 'adam', metrics = ['accuracy'])
model.load_weights('my_model_weights_0.h5')

predict = model.predict_classes(x_test)
with open(sys.argv[2],'w',newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['id','label'])
      for i in range(predict.shape[0]):
          writer.writerow(["%d"%(i),"%d"%predict[i]])