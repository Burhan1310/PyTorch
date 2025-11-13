import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # shape (n_samples, 1)
        self.n_samples = xy.shape[0]
        
        
    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]
        
    def __len__(self):
        # return length of dataset
        return self.n_samples
    
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(features, labels)