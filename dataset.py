import os
import glob
from collections import ChainMap
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class DrivingDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        imgs = glob.glob(f'{data_path}/*/CameraRGB/*.png')
        self.list_IDs = list(map(lambda s: s.replace(data_path, '').replace('.png', '')[1:], imgs))

        labels = glob.glob(f'{data_path}/*/*.pickle')
        label_dicts = list(map(lambda s: pickle.load(open(s, 'rb')), labels))
        self.labels = dict(ChainMap(*label_dicts)) 

        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        X = self.preprocess_x(ID)
        y = self.preprocess_y(ID)

        return X, y

    def preprocess_x(self, ID):
        image = Image.open(f'{self.data_path}/{ID}.png')
        if self.transform:
            return self.transform(image)

    def preprocess_y(self, ID):
        y = self.labels[ID]
        return torch.tensor([y['steer']])

