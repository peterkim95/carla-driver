#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


# In[2]:


import torch
import numpy as np
from pilotnet import PilotNet, get_transform
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F


# In[3]:


net = PilotNet()
net.load_state_dict(torch.load('../checkpoints/net_epoch_1.pt', map_location=torch.device('cpu')))


# In[4]:


for name, param in net.named_parameters():
    if param.requires_grad:
        print(name)


# In[5]:


net


# In[6]:


im = Image.open('../data/2020-10-14--09-02-25/train/episode_0003/RGBCenter/000140.png')
im


# In[7]:


t = get_transform()


# In[8]:


t(im).size()


# In[9]:


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# In[10]:


net.conv1.register_forward_hook(get_activation('conv1'))


# In[11]:


net.visual_mask.register_forward_hook(get_activation('visual_mask'))


# In[12]:


activation


# In[13]:


output = net(t(im).unsqueeze(0))


# In[17]:


conv1_act = activation['conv1']


# In[18]:


conv1_act.size()


# In[19]:


act = conv1_act.squeeze()


# In[20]:


act.size(0)


# In[114]:


F.relu(act[0])


# In[25]:


mt = transforms.Compose([transforms.CenterCrop((160,320)),transforms.Resize((66,200))])


# In[26]:


mt(im)


# In[21]:


r,c=6,4
fig, axarr = plt.subplots(r,c, sharex='all', sharey='all', figsize=(20, 10))
fig.subplots_adjust(hspace=0.0, wspace=0.0)
for i in range(r):
    for j in range(c):
        axarr[i][j].set_axis_off()
        axarr[i][j].imshow(act[4 * i + j], cmap='Greys')


# In[22]:


r,c=6,4
fig, axarr = plt.subplots(r,c, sharex='all', sharey='all', figsize=(20, 10))
fig.subplots_adjust(hspace=0.0, wspace=0.0)
for i in range(r):
    for j in range(c):
        axarr[i][j].set_axis_off()
        axarr[i][j].imshow(F.relu(act[4 * i + j]))


# In[39]:


plt.imshow(net.visual_mask.detach().squeeze())


# In[61]:


from matplotlib.colors import Normalize
x = plt.imshow(net.visual_mask.detach().squeeze(), norm=Normalize(vmin=0, vmax=1, clip=False), cmap='binary')


# In[62]:


mask = net.visual_mask.detach().squeeze().numpy()


# In[ ]:





# In[71]:


mask


# In[72]:


x = plt.imshow(mask, norm=Normalize(vmin=0, vmax=1, clip=False), cmap='binary')


# In[79]:


norm_mask = np.ma.getdata(Normalize(vmin=0, vmax=1, clip=True)(mask))


# In[81]:


np.ma.getmask(Normalize(vmin=0, vmax=1, clip=True)(mask))


# In[82]:


plt.imshow(norm_mask, cmap='binary')


# In[96]:


plt.imshow(mask, cmap='binary')


# In[92]:


norm_mask.shape


# In[99]:


net.visual_mask.detach().squeeze().numpy()


# In[98]:


norm_mask


# In[95]:


Image.fromarray(mask, 'RGBA')


# In[ ]:




