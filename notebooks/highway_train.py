#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


# In[2]:


sys.path.append("..") # ec2


# In[3]:


import torch
import numpy as np
from matplotlib import cm
from pilotnet import PilotNet, get_transform, get_truncated_transform
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F


# In[9]:


im = Image.open('../data/2020-11-06--00-54-17/train/episode_0002/CenterRGB/000140.png')
im


# In[12]:


t = get_transform()


# In[10]:


tt = get_truncated_transform()


# In[24]:


newt = transforms.Compose([
    transforms.CenterCrop((400, 320)),
])


# In[25]:


newt(im)


# In[11]:


tt(im)


# In[14]:


print(tt(im))


# In[15]:


oim = Image.open('../data/2020-10-14--09-02-25/train/episode_0003/RGBCenter/000140.png')
print(oim)


# In[9]:


print(im)


# In[13]:


t(im)


# In[ ]:




