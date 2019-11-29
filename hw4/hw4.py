import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import manifold, datasets
import sys




C=1
z_input=2048
hidden=12
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
 
        # define: encoder
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.encoder = nn.Sequential(#32*32
          nn.Conv2d(3, 32*C, 3, 2, 1), #16*16
          nn.SELU(), 

          nn.Conv2d(32*C,64*C, 3, 2, 1),#8*8
          nn.SELU(),

          nn.Conv2d(64*C,128*C, 3, 2, 1),  
          nn.SELU()

        )
        # define: decoder
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d( 128*C,64*C, 2, 2),
          nn.ConvTranspose2d(64*C,32*C, 2, 2),
          nn.ConvTranspose2d(32*C, 3, 2, 2),
          nn.Tanh()
        )
        self.linear_i=nn.Linear(z_input,hidden)
        self.linear_o=nn.Linear(hidden,z_input)
 
    def forward(self, x):
 
        encoded = self.encoder(x)
        encoded=self.linear_i(encoded.view(encoded.size(0), -1))
        #print(encoded.shape)
        hold=self.linear_o(encoded)
        decoded = self.decoder(hold.reshape(-1,128,4,4))
        return hold, decoded


if __name__ == '__main__':
 
    # detect is gpu available.
    use_gpu = torch.cuda.is_available()
 
    autoencoder = Autoencoder()
    
    # load data and normalize to [-1, 1]
    trainX = np.load(sys.argv[1])
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)
 
    # if use_gpu, send model / data to GPU.
    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()
 
    # Dataloader: train shuffle = True
    train_dataloader = DataLoader(trainX, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(trainX, batch_size=32, shuffle=False)
 
 
    # We set criterion : L1 loss (or Mean Absolute Error, MAE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
 
    # Now, we train 20 epochs.
    for epoch in range(40):
 
        cumulate_loss = 0
        for x in train_dataloader:
            
            latent, reconstruct = autoencoder(x)
            loss = criterion(reconstruct, x)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cumulate_loss = loss.item() * x.shape[0]
 
        print(f'Epoch { "%03d" % epoch }: Loss : { "%.5f" % (cumulate_loss / trainX.shape[0])}')
 
 
    # Collect the latents and stdardize it.
    latents = []
    reconstructs = []
    for x in test_dataloader:
 
        latent, reconstruct = autoencoder(x)
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())
 
    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)

    #latents = PCA(n_components=256).fit_transform(latents)
    # what the hell is tsne
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=8700)
    latents2 = latents
    latents = tsne.fit_transform(latents)
    # Use PCA to lower dim of latents and use K-means to clustering.
    
    result = KMeans(n_clusters = 2).fit(latents).labels_

    #what the hell is tsne
    
 
    # We know first 5 labels are zeros, it's a mechanism to check are your answers
    # need to be flipped or not.
    if np.sum(result[:5]) >= 3:
        result = 1 - result
 
 
    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(sys.argv[2],index=False)