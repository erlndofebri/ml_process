#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
# data dumb store
import joblib
# to locate yaml file
import yaml
# to locate directore
import os


# In[2]:


params_dir = "config/params.yaml"


# In[91]:


def load_params(param_dir):
    with open(param_dir, 'r') as file:
        params = yaml.safe_load(file)
        
    return params


# In[104]:


params = load_params(params_dir)


# In[85]:


params


# # Case

# - Objective: Bank wants to increase the number of deposite by optimizing data-driven marketing strategy
# - Concern: Short term revenue growth --> ML performance

# # Data Collection

# In[8]:


# fungsi read data csv
def read_data(path):
    data = pd.read_csv(path)
    return data


# In[116]:


path = 'data/awal/bank.csv'

data = read_data(path)
data


# In[163]:


data['pdays'][data['deposit']=="yes"].mean()


# In[162]:


data['pdays'][data['deposit']=="no"].mean()


# # Data Definition

# - Variable: Definition
# - ID: Unique client ID
# - age: Age of the client
# - job: Type of job
# - marital: Marital status of the client
# - eucation: Education level
# - default: Credit in default.
# - housing: Housing loan
# - loan: Personal loan
# - contact: Type of communication
# - month: Contact month
# - day_of_week: Day of week of contact
# - duration: Contact duration
# - campaign: number of contacts performed during this campaign to the client
# - pdays: number of days that passed by after the client was last contacted
# - previous: number of contacts performed before this campaign
# - poutcome: outcome of the previous marketing campaign
# 

# # Data Validation

# ## Data Type

# In[58]:


# Create Data Description

def data_desc(df):
    
    list_item = []
    for col in df.columns:
        list_item.append([col, df[col].dtype, 
                          df[col].isna().sum(),
                          100*df[col].isna().sum()/len(df[col]),
                          df[col].nunique(),
                          df[col].unique()[:4]])
        
    desc_df = pd.DataFrame(data=list_item, columns='feature data_type null_num null_pct unique_num unique_sample'.split())
    
    return desc_df


# In[59]:


# show general ata information
desc_df = data_desc(data)
desc_df


# ## Range

# In[60]:


data.describe()


# In[61]:


# ada balance negative --> diganti Nan
# ada pdays negative --> diganti Nan


# ## Data Dimenstion

# In[62]:


data.shape


# ## Handling `pdays` column

# In[117]:


# handling pdays function
def handling_negative(data, col):
    data[col] = np.where(data[col] < 0, -1, data[col])
    return data[col]


# In[118]:


data['pdays'] = handling_negative(data, 'pdays')


# ## Handling `balance` columns

# In[119]:


data['balance'] = handling_negative(data, 'balance')


# In[120]:


# describe data
data.describe()


# # Data Defense

# In[134]:


def check_data(input_data, params):
    # check data types
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object column(s)."
    assert input_data.select_dtypes("int").columns.to_list() == params["int32_columns"], "an error occurs in int32 column(s)."

    # check range of data
    assert set(input_data.job).issubset(set(params["range_job"])), "an error occurs in job range."
    assert set(input_data.marital).issubset(set(params["range_marital"])), "an error occurs in marital range."
    assert set(input_data.education).issubset(set(params["range_education"])), "an error occurs in education range."
    assert set(input_data.default).issubset(set(params["range_default"])), "an error occurs in default range."
    assert set(input_data.housing).issubset(set(params["range_housing"])), "an error occurs in housing range."
    assert set(input_data.loan).issubset(set(params["range_loan"])), "an error occurs in loan range."
    assert set(input_data.contact).issubset(set(params["range_contact"])), "an error occurs in contact range."
    assert set(input_data.month).issubset(set(params["range_month"])), "an error occurs in month range."
    assert set(input_data.poutcome).issubset(set(params["range_poutcome"])), "an error occurs in poutcome range."
    assert input_data.age.between(params["range_age"][0], params["range_age"][1]).sum() == len(input_data), "an error occurs in age range."
    assert input_data.balance.between(params["range_balance"][0], params["range_balance"][1]).sum() == len(input_data), "an error occurs in balance range."
    assert input_data.day.between(params["range_day"][0], params["range_day"][1]).sum() == len(input_data), "an error occurs in day range."
    assert input_data.duration.between(params["range_duration"][0], params["range_duration"][1]).sum() == len(input_data), "an error occurs in duration range."
    assert input_data.campaign.between(params["range_campaign"][0], params["range_campaign"][1]).sum() == len(input_data), "an error occurs in campaign range."
    assert input_data.pdays.between(params["range_pdays"][0], params["range_pdays"][1]).sum() == len(input_data), "an error occurs in pdays range."
    assert input_data.previous.between(params["range_previous"][0], params["range_previous"][1]).sum() == len(input_data), "an error occurs in previous range."


# In[141]:


check_data(data, params)


# In[129]:


params["int32_columns"]


# # Data Spliting

# In[142]:


X = data.drop('deposit', axis=1).copy()
y = data['deposit'].copy()


# In[143]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42,  stratify = y)


# In[144]:


x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42, stratify = y_test)


# In[145]:


x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape


# In[146]:


joblib.dump(x_train, "data/raw/x_train.pkl")
joblib.dump(y_train, "data/raw/y_train.pkl")
joblib.dump(x_valid, "data/raw/x_valid.pkl")
joblib.dump(y_valid, "data/raw/y_valid.pkl")
joblib.dump(x_test, "data/raw/x_test.pkl")
joblib.dump(y_test, "data/raw/y_test.pkl")


# In[ ]:




