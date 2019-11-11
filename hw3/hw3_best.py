# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:41:23 2019

@author: kevin
"""
import sys
import csv
import os
import random
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
def load_data(img_path, label_path):
    train_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    train_label = pd.read_csv(label_path)
    train_label = train_label.iloc[:,1].values.tolist()
    
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    train_set = train_data[:20000]
    valid_set = train_data[20000:]
    
    return train_set, valid_set
#Class from TA
class hw3_dataset(Dataset):
    
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0])
        img = self.transform(img)
        label = self.data[idx][1]
        return img, label
#I create a calss for test data myself, easier to predict as far as I know
class hw3_testset(Dataset):
    
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        img = self.transform(img)
        return img

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),     
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        #image size (48,48)
        x = self.conv1(x) #(24,24)
        x = self.conv2(x) #(12,12)
        x = self.conv3(x) #(6,6)
        x = self.conv4(x) #(3,3)
        x = x.view(-1, 3*3*128)
        x = self.fc(x)
        return x

#train_set, valid_set = load_data('train_img/train_img/train_img/', 'train.csv')
transform = transforms.Compose([
    #transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([mean], [std], inplace=False)
    ])
    
#train_dataset = hw3_dataset(train_set,transform)
#train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

#valid_dataset = hw3_dataset(valid_set,transform)
#valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

model = Net()
model.load_state_dict(torch.load('model_31.pth'))

##load test data
#test_img_path='/content/test_img/'
test_image = sorted(glob.glob(os.path.join(sys.argv[1], '*.jpg')))
test_data = list(test_image)
test_dataset = hw3_testset(test_data,transform)
test_loader =  DataLoader(test_dataset)
result=[]
model.eval()
for idx, img in enumerate(test_loader):
     output = model(img)
     predict = torch.max(output, 1)[1].data.numpy()
     result.append(predict)


#write csv
with open(sys.argv[2],'w',newline='') as csvfile:
     writer = csv.writer(csvfile)
     writer.writerow(['id','label'])
     for i in range(len(result)):
          writer.writerow(["%d"%(i),"%d"%result[i][0]])   
     