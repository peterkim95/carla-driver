import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class DrivingDataset(Dataset):
    def __init__(self, list_IDs, labels, transform=None):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        X = self.preprocess_x(ID)
        y = self.preprocess_y(ID)

        return X, y

    def preprocess_x(self, ID):
        image = Image.open(f'data/2020-23-09_07:52:54/episode_0000/CameraRGB/{ID}.png')
        if self.transform:
            return self.transform(image)

    def preprocess_y(self, ID):
        y = self.labels[ID]
        return torch.tensor([y['steer']])


def generate_partition():
    files = os.listdir('data/2020-23-09_07:52:54/episode_0000/CameraRGB')
    ids = list(map(lambda s: s.split('.')[0], files))
    np.random.shuffle(ids)

    sample_size = len(ids)
    i = round(sample_size * 0.8)
    return {'train': ids[:i], 'val': ids[i:]}


def generate_labels():
    with open('data/2020-23-09_07:52:54/episode_0000/label.pickle', 'rb') as f:
        return pickle.load(f)

