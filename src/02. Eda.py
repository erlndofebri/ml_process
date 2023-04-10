#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
import scipy.stats as scs
# data dumb store
import joblib
# to locate yaml file
import yaml
# to locate directore
import os


# # Case

# - Objective: Bank wants to increase the number of deposite by optimizing data-driven marketing strategy
# - Concern: Long term revenue growth --> Interpretability

# # Load Training Dataset

# In[2]:


x_train = joblib.load("data/raw/x_train.pkl")
y_train = joblib.load("data/raw/y_train.pkl")


# In[3]:


dataset = pd.concat([x_train, y_train], axis = 1)
dataset


# # Exploratory Data Anlaysis

# ## Check Missing Values

# In[4]:


dataset.isnull().sum()


# ## Check Stats

# In[5]:


dataset.skew()


# In[6]:


# almost all continous data is skew


# In[7]:


dataset.describe()


# ## Split by Class

# In[8]:


dataset_deposit = dataset[dataset['deposit']=="yes"]
dataset_no_deposit = dataset[dataset['deposit']=="no"]


# In[9]:


dataset_deposit.describe().T[["mean", "50%"]]


# In[10]:


dataset_no_deposit.describe().T[["mean", "50%"]]


# In[11]:


# balance diantara 2 kelas memiliki perbedaan mean yang besar
# pdays diantara 2 kelas memiliki perbedaan mean yang besar
# duration diantara 2 kelas memiliki perbedaan mean yang besar


# ## Kdeplot

# In[12]:


num = dataset.select_dtypes(include="number").columns


# In[13]:


plt.figure(figsize=(15,10))
for i in range(0, len(num)):
    plt.subplot(3,3,i+1)
    sns.kdeplot(x=dataset[num[i]], hue=dataset['deposit'])
    plt.title(num[i], fontsize=15)
    plt.xlabel(' ')
    plt.tight_layout()


# ## Boxplot

# In[14]:


plt.figure(figsize=(15,10))
for i in range(0, len(num)):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=dataset[num[i]], palette='Blues_d')
    plt.title(num[i], fontsize=15)
    plt.xlabel(' ')
    plt.tight_layout()


# - We'll try to use original data instead of removing the outliers in the first trial

# ## Countplot

# In[15]:


cat = dataset.select_dtypes(include='object').columns


# In[16]:


len(cat)


# In[17]:


fig, axes = plt.subplots(4,3, figsize=(15,12))
xy = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2), (3,0)]

for i in range(len(cat)):
    sns.countplot(y=dataset[cat[i]], palette='viridis', ax=axes[xy[i]])
    axes[3,1].axis('off')
    axes[3,2].axis('off')
    plt.tight_layout() 


# - FP --> predict deposite, alsinya gak
# - FN --> predict no, aslinya deposit
# 
# - yg diprediksi depoite --> dikasih treatment marketing biar makin yaquin deposit
# - FP tinggi, marketingnya boncos
# - FN tinggi, kehilangan potensi customer --> berat disini (recall)
# 
# - recall:
# - TP/ FN + TP

# ## Correlation plot

# In[27]:


data_corr = dataset.copy()
data_corr['deposit'] = np.where(data_corr['deposit'] == "no", 0,1)


# In[28]:


data_corr['deposit'].unique()


# In[29]:


# Create Heatmap Correlation Chart for Each Numerical Features

plt.figure(figsize=(12, 12))
sns.heatmap(data_corr.corr(), cmap='Blues', annot=True, fmt='.2f')


# ## Kesimpulan

# - negative value `balance` diimputasi dgn median masing2 kelas
# - negative value `pdays` di imputasi dgn 0
