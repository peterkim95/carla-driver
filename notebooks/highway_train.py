#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


# In[10]:


import torch
import numpy as np
from matplotlib import cm
from pilotnet import PilotNet, get_transform, get_truncated_transform
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F


# In[3]:


im = Image.open('../data/2020-10-29--14-46-09/train/episode_0000/CenterRGB/000140.png')
im


# In[4]:


t = get_transform()


# In[11]:


tt = get_truncated_transform()


# In[14]:


print(tt(im))


# In[15]:


oim = Image.open('../data/2020-10-14--09-02-25/train/episode_0003/RGBCenter/000140.png')
print(oim)


# In[9]:


print(im)


# In[5]:


t(im)


# In[ ]:




