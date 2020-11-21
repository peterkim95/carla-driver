import os
import glob
from collections import ChainMap
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class DrivingDataset(Dataset):
    def __init__(self, timestamps, transform=None):
        self.data_path = 'data'
        self.list_IDs = []
        label_dicts = []
        for ts in timestamps:
            imgs = glob.glob(f'data/{ts}/*/*/*.png') # e.g. data_path/episode_x/RGBCenter/x.png
            self.list_IDs += list(map(lambda s: s.replace(self.data_path, '').replace('.png', '')[1:], imgs))

            labels = glob.glob(f'data/{ts}/*/*.pickle')
            label_dicts += list(map(lambda s: pickle.load(open(s, 'rb')), labels))

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
        image = Image.open(f'{self.data_path}/{ID}.png').convert('RGB')
        if self.transform:
            return self.transform(image)

    def preprocess_y(self, ID):
        y = self.labels[ID]
        return torch.tensor([y['steer'], sig_to_tanh(y['throttle']), sig_to_tanh(y['brake'])])


def tanh_to_sig(x):
    return x

def sig_to_tanh(x):
    return x

