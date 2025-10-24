#!/usr/bin/env python
# coding: utf-8

# The following is an image processing technique used to track and determine characteristics of cell movement. 

# ### Setup the Project:

# In[1]:


from __future__ import division, unicode_literals, print_function
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

mpl.rc("figure", figsize=(10, 5))
mpl.rc('image', cmap='gray')


# In[2]:


import numpy as np 
import pandas as pd
from pandas import DataFrame, Series


# In[3]:


import pims
import trackpy as tp
import collections


# ### Read data and convert to grayscale

# In[4]:


@pims.pipeline
def gray(image):
    return image[:, :, 1]

video_path = '/Users/ashty/Downloads/testcells.avi'

if os.path.exists(video_path):
    frames = gray(pims.Video(video_path))
    print('Video loaded successfully!')
else: 
    print('Video fild not found:', video_path)
    


# In[5]:


frames


# #### Show the first frame

# In[6]:


plt.imshow(frames[0]);


# ### Locate cells

# In[7]:


f = tp.locate(frames[0], 17, invert=True)


# In[8]:


f.head()


# In[9]:


tp.annotate(f, frames[0]);


# ### Filter data

# In[10]:


# show the mass(darkness intensity) which includes noise
fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)

ax.set(xlabel='mass', ylabel='count');


# In[11]:


# Set the minmass parameter based on the prior data
f = tp.locate(frames[0], 17, invert=True, minmass=1000)
tp.annotate(f, frames[0]);


# ### Check for Subpixel Accuracy 

# In[12]:


tp.subpx_bias(f)


# In[13]:


tp.subpx_bias(tp.locate(frames[0], 17, invert=True, minmass=1000));


# ### Collect data from set of frames into a table

# In[14]:


f = tp.batch(frames[:150], 17, minmass=1000, invert=True,processes=1);


# ### Link located cells into particle trajectories

# In[15]:


t = tp.link(f, 10, memory=3)


# In[16]:


t.head()


# ### Filter Trajectories

# #### Keep only the trajectories that last a certain number of frames (here we test 10 frames)

# In[17]:


t1 = tp.filter_stubs(t, 10)


# #### Compare the # of trajectories before and after filtering

# In[18]:


print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())


# #### Can also filter by particles' appearance

# In[19]:


# Plot mass vs. size of particles
plt.figure()
tp.mass_size(t1.groupby('particle').mean());


# #### Now apply the filter based on the mass, size, and eccentricity

# In[20]:


t2 = t1[((t1['mass'] > 1000) & (t1['size'] < 4) & (t1['ecc'] < 3))]


# ### Plot the filtered trajectories

# In[21]:


plt.figure()
tp.annotate(t2[t2['frame'] == 0], frames[0])


# In[22]:


plt.figure()
tp.plot_traj(t2)


# In[23]:


# Save the final tracking data
t2.to_excel('TrajectoryData.xlsx', index=False)


# ## Find Average Cell Position Every Third Frame

# In[24]:


t3=t2.copy()


# In[25]:


df = 3
t3['x_ave'] = (t3['x'] + t3.groupby('particle')['x'].shift(1) + t3.groupby('particle')['x'].shift(2)) / df
t3['y_ave'] = (t3['y'] + t3.groupby('particle')['y'].shift(1) + t3.groupby('particle')['y'].shift(2)) / df
t4 = t3[(t3['frame']+1) % 3 == 0]
t5 = (t4[['frame', 'particle', 'x_ave', 'y_ave']])
print(t5)


# In[26]:


t5.to_excel('AveragePosition.xlsx', index=False)


# ## Find Velocity (pixels/second) & Angle

# #### Here we have 30 frames/second and we took the average position of three frames therefore dt = 1/30 * 3 = 0.1second

# In[27]:


t6 = t5.copy()


# In[28]:


dt= 0.1
t6['dx']=t6['x_ave'] - t6.groupby('particle')['x_ave'].shift(1)
t6['dy']=t6['y_ave'] - t6.groupby('particle')['y_ave'].shift(1)
t6['velocity'] = ((t6['dx'] ** 2 + t6['dy'] ** 2) ** 0.5) / dt

t6['movement_angle'] = np.arctan2(t6['dy'], t6['dx']) * (180/np.pi)


# In[29]:


v = (t6[t6['frame'] >=5])
print(v)


# In[30]:


v.to_excel('Cell_Velocity_And_Angle.xlsx', index = False)


# In[31]:


fig, ax = plt.subplots()
ax.hist(v['velocity'], bins =20)
ax.set(xlabel='velocity', ylabel='count');


# #### Calculate the average velocity for each particle

# In[ ]:


#v1=v.copy()
#dframe= would have to know number of frames which is tricky since some particles dont last through all frames

#v1['ave_velocity']= (v1.groupby('particle')['velocity'].sum()) / dframe


# In[ ]:




