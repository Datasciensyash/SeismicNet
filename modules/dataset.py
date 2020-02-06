from torch.utils.data import Dataset
import torch
import random
import numpy as np

class SeismicDataset(Dataset):
    def __init__(self, train_X, train_Y, aug, dtype='Train', normalize=True, reverse_chance=0.1):

        self.train_X = train_X.astype('float32')
        self.train_Y = train_Y.astype('float32')
        self.aug = aug
        self.dtype = dtype
        self.reverse_chance = reverse_chance

        if normalize:
            self.normalize()

    def normalize(self):
        """
        Normalizes data by substracting its mean and divide by std.
        """
        self.train_X = (self.train_X - self.train_X.mean()) / self.train_X.std()

    def swap_axes(self):
        """
        Changing type of section:
            Inline -> Crossline
            Crossline -> Inline
        """
        self.train_X = np.swapaxes(self.train_X, 0, 1)
        self.train_Y = np.swapaxes(self.train_Y, 0, 1)        
        
    def __len__(self):


        if if dtype == 'Train' and random.random() < self.reverse_chance:
            self.swap_axes()
            
        return len(self.train_Y)
    
    def __getitem__(self, idx):
        if self.dtype == 'Train':
            augmented = self.aug(image=self.train_X[idx], mask=self.train_Y[idx])
            img, mask = torch.Tensor(augmented['image']), torch.Tensor(augmented['mask'])  
            return img, mask

        elif self.dtype == 'Test':
            img, mask = torch.Tensor(self.train_X[idx]), torch.Tensor(self.train_Y[idx])
            return img, mask