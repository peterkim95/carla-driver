#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import sys


# In[8]:


# TODO: sys.path hack to call stuff up a directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))


# In[28]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[7]:


from dataset import DrivingDataset


# In[12]:


train_path, val_path = '../data/2020-10-03_14:11:32/train', '../data/2020-10-03_14:11:32/val'


# In[13]:


train_set = DrivingDataset(train_path)


# In[14]:


train_set.labels


# In[15]:


label_df = pd.DataFrame.from_dict(train_set.labels, orient='index')


# In[19]:


label_df['steer']


# In[34]:


max(label_df['steer']), min(label_df['steer'])


# In[32]:


sns.displot(label_df['steer'], bins=np.arange(-1, 1.001, 0.05))

