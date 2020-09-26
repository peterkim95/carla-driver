import os
import pickle

import torch
import numpy as np
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        X = self.preprocess_x(ID)
        y = self.preprocess_y(ID)

        return X, y

    def preprocess_x(self, ID):
        image = Image.open(f'data/2020-23-09_07:52:54/episode_0000/CameraRGB/{ID}.png')
        return torch.from_numpy(np.asarray(image).copy().astype('float32') / 255.0).permute([2,1,0]) # (3, 800, 600)

    def preprocess_y(self, ID):
        y = self.labels[ID]
        return torch.tensor([y['steer']])


def generate_partition():
    files = os.listdir('data/2020-23-09_07:52:54/episode_0000/CameraRGB')
    ids = list(map(lambda s: s.split('.')[0], files))
    np.random.shuffle(ids)

    sample_size = len(ids)
    i = round(sample_size * 0.8)
    return {'train': ids[:i], 'validation': ids[i:]}


def generate_labels():
    with open('data/2020-23-09_07:52:54/episode_0000/label.pickle', 'rb') as f:
        return pickle.load(f)

