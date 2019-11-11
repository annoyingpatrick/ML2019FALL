# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:20:46 2019

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

    use_gpu = torch.cuda.is_available()
    train_set, valid_set = load_data(sys.argv[1], sys.argv[2])
    transform = transforms.Compose([
    #transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([mean], [std], inplace=False)
    ])
    train_dataset = hw3_dataset(train_set,transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataset = hw3_dataset(valid_set,transform)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    model = Net()
    if use_gpu:
        print("use gpu")
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    num_epoch = 20
    train_loss_his = []
    train_acc_his = []
    valid_loss_his = []
    valid_acc_his = []
    
    
    for epoch in range(num_epoch):
        model.train()
        train_loss = []
        train_acc = []
        counter=0
        for idx, (img, label) in enumerate(train_loader):
            counter=counter+1
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
            if (counter%50==0):
              print("epoch%d_%s"%(epoch,counter))
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
        train_loss_his.append(np.mean(train_loss))
        train_acc_his.append(np.mean(train_acc))

        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for idx, (img, label) in enumerate(valid_loader):
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
            valid_loss_his.append(np.mean(valid_loss))
            valid_acc_his.append(np.mean(valid_acc))
        
        if np.mean(train_acc) > 0.9:
            checkpoint_path = 'model_{}.pth'.format(epoch+1) 
            torch.save(model.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)