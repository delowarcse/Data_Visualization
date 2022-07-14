#!/usr/bin/env python
# coding: utf-8

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
stroke_data = pd.read_csv('Injured Participant Data.csv', delim_whitespace=False)
control_data = pd.read_csv('Healthy Control Participant Data.csv', delim_whitespace=False)

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


# Descriptive Statistics
subset_attributes = ['Abs error X', 'Abs error Y', 'Abs error XY', 'Variability X', 'Variability Y', 'Variability XY', 'Contraction expansion ratio X', 'Contraction expansion ratio Y', 'Contraction expansion ratio X', 'Shift X','Shift Y','Shift XY']
sd = round(stroke_data[subset_attributes].describe(),2)
cd = round(control_data[subset_attributes].describe(),2)
pd.concat([cd, sd], axis=0, keys=['Control Data Statistics', 'Stroke Data Statistics'])


# In[ ]:


# Visualizing one dimension
datas.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
              xlabelsize=9, ylabelsize=9, grid=False)    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))


# In[ ]:


import matplotlib.ticker as mtick

fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Abs error X in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Abs error X', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465)) #xmax=429
ax.set_xlabel("Abs error X (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#fig.savefig('Abs err X (percentage).png',bbox_inches='tight')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Abs error Y in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Abs error Y', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Abs error Y (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Abs error XY in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Abs error XY', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Abs error XY (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Variability X in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Variability X', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Variability X (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Variability Y in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Variability Y', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

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


plt.savefig('Varibaility Y.eps', format='eps')


# In[ ]:


fig1.savefig('Varibaility_Y_04032022_300dpi.jpg', dpi=300)


# In[ ]:


fig1.savefig('Varibaility_Y_04032022_600dpi.jpg', dpi=600)


# In[ ]:


fig1.savefig('Varibaility_Y_04032022_1200dpi.jpg', dpi=1200)


# In[ ]:


fig1.savefig('Varibaility_Y_04032022_3000dpi.jpg', dpi=3000)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Variability Y in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Variability Y', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

#ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Variability Y (m)")
ax.set_ylabel("Participants")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#fig.savefig('Variability (percentage).png',bbox_inches='tight')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Variability XY in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Variability XY', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Variability XY (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Contraction expansion ratio X in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Contraction expansion ratio X', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Contraction expansion ratio X")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Contraction expansion ratio Y in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Contraction expansion ratio Y', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Contraction expansion ratio Y")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Contraction expansion ratio XY in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Contraction expansion ratio XY', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Contraction expansion ratio XY")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Shift X in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Shift X', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Shift X (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Shift X in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Shift X', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Shift Y (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Shift XY in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Shift XY', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=465))
ax.set_xlabel("Shift XY (m)")
ax.set_ylabel("Participants (%)")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:




