#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys


# In[2]:


sys.path.append("..") # Adds higher directory to python modules path. # Works for EC2


# In[12]:


import pandas as pd
import seaborn as sns
import numpy as np
import torchvision.transforms as transforms
import torch


# In[26]:


from pilotnet import PilotNet
from dataset import DrivingDataset


# In[6]:


data_path = '2020-10-04_06:22:49'


# In[58]:


val_path = f'../data/{data_path}/val'
train_path = f'../data/{data_path}/train'


# In[9]:


transform = transforms.Compose([transforms.Resize((200, 66)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# In[55]:


val_set = DrivingDataset(val_path, transform)


# In[56]:


val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, batch_size=32)


# In[59]:


train_set = DrivingDataset(train_path, transform)


# In[60]:


train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=32)


# In[35]:


label_df = pd.DataFrame.from_dict(val_set.labels, orient='index')


# In[36]:


label_df[:100]


# In[25]:


label_df['throttle'].value_counts()


# In[66]:


net = PilotNet()
net.load_state_dict(torch.load('../checkpoints/net_epoch_0.pt'))


# # epoch 20 train

# In[65]:


with torch.no_grad():
    for x, y in train_loader:
        output = net(x)

        r = pd.DataFrame({'y_true': y.squeeze().tolist(), 'y_pred': output.squeeze().tolist()})
        r['diff'] = abs(r['y_true'] - r['y_pred'])
        print(r)
        break


# # epoch 20 val

# In[64]:


with torch.no_grad():
    for x, y in val_loader:
        output = net(x)

        r = pd.DataFrame({'y_true': y.squeeze().tolist(), 'y_pred': output.squeeze().tolist()})
        r['diff'] = abs(r['y_true'] - r['y_pred'])
        print(r)
        break


# # bad model train

# In[67]:


with torch.no_grad():
    for x, y in train_loader:
        output = net(x)

        r = pd.DataFrame({'y_true': y.squeeze().tolist(), 'y_pred': output.squeeze().tolist()})
        r['diff'] = abs(r['y_true'] - r['y_pred'])
        print(r)
        break


# # bad model val

# In[68]:


with torch.no_grad():
    for x, y in val_loader:
        output = net(x)

        r = pd.DataFrame({'y_true': y.squeeze().tolist(), 'y_pred': output.squeeze().tolist()})
        r['diff'] = abs(r['y_true'] - r['y_pred'])
        print(r)
        break


# In[ ]:




