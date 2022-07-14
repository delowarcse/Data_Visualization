#!/usr/bin/env python
# coding: utf-8

# # This program is the implementation of the data visualization of two varibales with KDE and percentage plot

# In[ ]:


# Import necessary dependencies
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Load and merge datasets # white = control; red = stroke; wine = data
stroke_data = pd.read_csv('Injured Participants Data.csv', delim_whitespace=False)
control_data = pd.read_csv('Healthy Control.csv', delim_whitespace=False)

# store wine type as an attribute
stroke_data['data_type'] = 'stroke'   
control_data['data_type'] = 'control'

# merge control and stroke data
datas = pd.concat([stroke_data, control_data])
datas = datas.sample(frac=1, random_state=42).reset_index(drop=True)

# understand dataset features and values
datas.head()
#stroke_data.head()
#control_data.head()


# In[ ]:


#plt.close(2)
fig1 = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Variability Y in Data", fontsize=14)
fig1.subplots_adjust(top=0.85, wspace=0.3)
ax = fig1.add_subplot(1,1, 1)

g1 = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "fuchsia", "control": "black"})
g1.map(sns.distplot, 'Variability Y', kde=False, bins=15, ax=ax)
ax.legend(title='Data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Variability Y (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#fig.savefig('Variability (percentage).png',bbox_inches='tight')
plt.close(2)


# In[ ]:


import matplotlib.ticker as mtick
fig = plt.figure(figsize = (10, 4))
#title = fig.suptitle("Variability X (m) in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

fig, ax = plt.subplots()
sns.kdeplot(stroke_data['Variability Y'], ax=ax, shade=True, color='fuchsia', label='Stroke')
sns.kdeplot(control_data['Variability Y'], ax=ax, shade=True, color='black', label='Control')
ax.legend(title='Data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465)) #xmax=429
ax.set_xlabel("Variability Y (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#fig.savefig('Abs err X (percentage).png',bbox_inches='tight')
#plt.close(2)


# In[ ]:


fig.savefig('Varibaility_Y_11032022_300dpi.jpg', dpi=300)


# In[ ]:


fig.savefig('Varibaility_Y_11032022_600dpi.jpg', dpi=600)


# In[ ]:


fig.savefig('Varibaility_Y_11032022_1200dpi.jpg', dpi=1200)


# In[ ]:


fig.savefig('Varibaility_Y_11032022_3000dpi.jpg', dpi=3000)


# In[ ]:


import matplotlib.ticker as mtick
fig = plt.figure(figsize = (10, 4))
#title = fig.suptitle("Variability X (m) in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

fig, ax = plt.subplots()
sns.kdeplot(stroke_data['Variability Y'], ax=ax, shade=True, color='fuchsia', label='Stroke')
sns.kdeplot(control_data['Variability Y'], ax=ax, shade=True, color='black', label='Control')
ax.legend(title='Data Type')

#ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465)) #xmax=429
ax.set_xlabel("Variability Y (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')


# In[ ]:




