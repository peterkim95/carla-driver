import os
import errno
from argparse import ArgumentParser

import torch

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


def get_args():
    parser = ArgumentParser(description='trainer')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints')
    parser.add_argument('--train', type=str)
    parser.add_argument('--val', type=str)

    args = parser.parse_args()
    return args


def save_checkpoint(state, filename='checkpoints/checkpoint.pt'):
    torch.save(state, filename)
