#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys


# In[2]:


sys.path.append("..") # Adds higher directory to python modules path. # Works for EC2


# In[ ]:


# TODO: sys.path hack to call stuff up a directory 
# This works for Mac
sys.path.insert(1, os.path.join(sys.path[0], '..'))


# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[4]:


from dataset import DrivingDataset


# In[5]:


data_path = '2020-10-04_06:22:49'


# In[6]:


train_path, val_path = f'../data/{data_path}/train', f'../data/{data_path}/val'


# In[12]:


train_set, val_set = DrivingDataset(train_path), DrivingDataset(val_path)


# In[ ]:


train_set.labels


# In[8]:


label_df = pd.DataFrame.from_dict(train_set.labels, orient='index')


# In[ ]:


label_df['steer']


# In[9]:


max(label_df['steer']), min(label_df['steer'])


# In[10]:


sns.displot(label_df['steer'], bins=np.arange(-1, 1.001, 0.05))


# In[13]:


label_df = pd.DataFrame.from_dict(val_set.labels, orient='index')


# In[14]:


max(label_df['steer']), min(label_df['steer'])


# In[15]:


sns.displot(label_df['steer'], bins=np.arange(-1, 1.001, 0.05))


# In[ ]:




