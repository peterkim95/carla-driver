import os
import shutil
import numpy as np
import errno
from argparse import ArgumentParser
from datetime import datetime

import torch

def get_current_datetime():
    return datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

def makedirs(name):
    try:
        os.makedirs(name)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise

def split_data(data_path, max_episodes, split_ratio):
    makedirs(f'{data_path}/train')
    makedirs(f'{data_path}/val')
    
    episodes = np.arange(max_episodes)
    np.random.shuffle(episodes)
    idx = int(max_episodes * split_ratio)
    train, val = episodes[:idx], episodes[idx:]

    for e in train:
        shutil.move(f'{data_path}/episode_{e:0>4d}', f'{data_path}/train/episode_{e:0>4d}')
    for e in val:
        shutil.move(f'{data_path}/episode_{e:0>4d}', f'{data_path}/val/episode_{e:0>4d}')

def get_args():
    parser = ArgumentParser(description='trainer')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--checkpoints_path', type=str, default='checkpoints')
    parser.add_argument('--train', type=str)
    parser.add_argument('--val', type=str)

    args = parser.parse_args()
    return args


def save_checkpoint(state, filename='checkpoints/checkpoint.pt'):
    torch.save(state, filename)
