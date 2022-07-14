#!/usr/bin/env python
# coding: utf-8

# # This program is the data visualization with Individual variable plots

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
#datas['quality_label'] = datas['Abs error XY'].apply(lambda value: 'low' if value < 0.02 else 'medium' if value < 0.04 else 'high')

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


# Visualizing two dimensions
f, ax = plt.subplots(figsize=(11, 8))
corr = datas.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
            linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Arm Position Matching Attributes Correlation Heatmap', fontsize=14)


# In[ ]:


cols = ['Abs error X', 'Abs error Y', 'Abs error XY', 'Variability X', 'Variability Y', 'Variability XY', 'Shift X','Shift Y','Shift XY']
pp = sns.pairplot(datas[cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Arm Position Matching Attributes Pairwise Plots', fontsize=14)


# In[ ]:


cols = ['Abs error X', 'Abs error Y', 'Abs error XY', 'Variability X', 'Variability Y', 'Variability XY', 'Contraction expansion ratio X', 'Contraction expansion ratio Y', 'Contraction expansion ratio X', 'Shift X','Shift Y','Shift XY']
subset_df = datas[cols]

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)
scaled_df.reset_index(drop=True, inplace=True)
datas.reset_index(drop=True, inplace=True)
final_df = pd.concat([scaled_df, datas['data_type']], axis=1)
final_df.head()


# In[ ]:


from pandas.plotting import parallel_coordinates

pc = parallel_coordinates(final_df, 'data_type', color=('#FFE888', '#FF9999'))


# Two Continuous Numeric attributes

# In[ ]:


# Two Continuous Numeric attributes
plt.scatter(datas['Abs error X'], datas['Abs error Y'],
            alpha=0.4, edgecolors='w')

plt.xlabel('Abs error X')
plt.ylabel('Abs error Y')
plt.title('Abs Error',y=1.05)


# In[ ]:


jp = sns.jointplot(x='Abs error X', y='Abs error Y', data=datas,
              kind='reg', space=0, size=5, ratio=4)


# Mixed attributes (numeric & categorical)

# In[ ]:


fig = plt.figure(figsize = (10,4))
title = fig.suptitle("Abs Error in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(1,2, 1)
ax1.set_title("Control Data")
ax1.set_xlabel("Abs error X")
ax1.set_ylabel("Frequency") 
ax1.set_ylim([0, 200])
ax1.text(1.2, 200, r'$\mu$='+str(round(control_data['Abs error X'].mean(),2)), fontsize=12)
r_freq, r_bins, r_patches = ax1.hist(control_data['Abs error X'], color='red', bins=15,
                                     edgecolor='black', linewidth=1)

ax2 = fig.add_subplot(1,2, 2)
ax2.set_title("Stroke Data")
ax2.set_xlabel("Abs error X")
ax2.set_ylabel("Frequency")
ax2.set_ylim([0, 200])
ax2.text(0.8, 200, r'$\mu$='+str(round(stroke_data['Abs error X'].mean(),2)), fontsize=12)
w_freq, w_bins, w_patches = ax2.hist(stroke_data['Abs error X'], color='white', bins=15,
                                     edgecolor='black', linewidth=1)


# In[ ]:


fig = plt.figure(figsize = (10, 4))
title = fig.suptitle("Abs Error in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(1,2, 1)
ax1.set_title("Control Data")
ax1.set_xlabel("Abs error X")
ax1.set_ylabel("Density") 
sns.kdeplot(control_data['Abs error X'], ax=ax1, shade=True, color='r')

ax2 = fig.add_subplot(1,2, 2)
ax2.set_title("Stroke Data")
ax2.set_xlabel("Abs error X")
ax2.set_ylabel("Density") 
sns.kdeplot(stroke_data['Abs error X'], ax=ax2, shade=True, color='y')


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Abs Error X in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Abs error X")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Abs error X', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Abs Error Y in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Abs error Y")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Abs error Y', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Abs Error XY in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Abs error XY")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Abs error XY', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Variability X in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Variability X")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Variability X', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Variability Y in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Variability X")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Variability Y', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Variability XY in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Variability XY")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Variability XY', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Contraction expansion ratio X in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Contraction expansion ratio X")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Contraction expansion ratio X', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Contraction expansion ratio Y in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Contraction expansion ratio Y")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Contraction expansion ratio Y', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Contraction expansion ratio XY in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Contraction expansion ratio XY")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Contraction expansion ratio XY', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Shift X in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Shift X")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Shift X', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Shift Y in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Shift Y")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Shift Y', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Shift XY in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Shift XY")
ax.set_ylabel("Frequency") 

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Shift XY', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')
plt.close(2)


# In[ ]:


import matplotlib.ticker as mtick

fig = plt.figure(figsize = (6, 4))
#title = fig.suptitle("Shift XY in Data", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1, 1)

g = sns.FacetGrid(datas, hue='data_type', palette={"stroke": "r", "control": "g"})
g.map(sns.distplot, 'Shift XY', kde=False, bins=15, ax=ax)
ax.legend(title='data Type')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=170))
ax.set_xlabel("Shift XY [m]")
#ax.set_ylabel("Frequency")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.close(2)


# In[ ]:




