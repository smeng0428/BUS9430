#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 


# In[2]:


raw_data = pd.read_excel('https://raw.githubusercontent.com/smeng0428/BUS9430/main/loan.xlsx')


# In[3]:


raw_data.head()


# In[4]:


raw_data.shape


# In[5]:


raw_data.info


# In[6]:


## get the description for the features.

codebook = pd.read_excel('https://raw.githubusercontent.com/smeng0428/BUS9430/main/Data_Dictionary.xlsx', sheet_name='LoanStats')


# In[7]:


## clean up the code book.the last two raws(115,116) are just footnotes.
codebook = codebook[:114].copy()


# In[8]:


codebook = codebook.rename(columns = {'LoanStatNew': "Feature"})
codebook.head()


# ## split the dataset to training/validation set

# In[9]:


train_set, val_set = train_test_split(raw_data, train_size = 0.7, random_state = 9340)


# In[10]:


## check for missing data
pd.set_option('display.float_format', lambda x: '%.5f' % x)
missing_ratios = raw_data.isna().sum() / raw_data.shape[0]
missing_ratios.sort_values(ascending = False)


# In[11]:


## significant part of the feature have nearly all missing values. These features does not provide much info.
## small portion of features have 30-70% missing values.
missing_ratios.hist()


# In[12]:


## append the disctription of features to a df
unique_ns =[]
unique_values = []
na_percent = train_set.isna().sum()/train_set.shape[0]
for i in train_set.columns:
    unique_ns.append(len(train_set[i].unique()))
    unique_values.append(train_set[i].unique())

tmp = pd.DataFrame(list(zip(train_set.columns,unique_ns, unique_values, na_percent)), columns=['Feature', 'unique_counts', 'unique_values','na_percent'])    
data_dictionary = pd.merge(tmp, codebook, how='left', on = "Feature")
data_dictionary


# In[13]:


##set missing ratio 0.8 as a threshold
data_dictionary[data_dictionary.na_percent<0.8].shape[0]

## this method reduce the features to 55


# In[14]:


## remove unwantted feature from training and validation set
train_set = train_set[data_dictionary[data_dictionary.na_percent<0.8].Feature]
val_set = val_set[data_dictionary[data_dictionary.na_percent<0.8].Feature]


# In[15]:


# features with no missing values
data_dictionary[data_dictionary.na_percent==0].shape[0]

## 43 features does not have missing values


# In[16]:


## features need to be impute
data_dictionary[(data_dictionary.na_percent>0) & (data_dictionary.na_percent<0.8)]
## some of the features need nlp or other techiniques to process fisrt.
## 'collections_12_mths_ex_med','chargeoff_within_12_mths','tax_liens' needs to be droped because most values are 0


# In[17]:


## create a list for features need to be dropped 
## Id and member_id doesn't help with modeling. These two needs to be droped too
drop_list = ['id', 'member_id', 'collections_12_mths_ex_med','chargeoff_within_12_mths','tax_liens','url']


# In[18]:


## drop features from train and val sets
for i in drop_list:
    del train_set[i]
    del val_set[i]


# In[19]:


## impute 'mths_since_last_delinq' with median 
train_set.mths_since_last_delinq.describe()


# In[20]:


train_set[train_set.mths_since_last_delinq.isna()]


# In[21]:


train_set.mths_since_last_delinq.fillna(train_set.mths_since_last_delinq.median(), inplace=True)
val_set.mths_since_last_delinq.fillna(train_set.mths_since_last_delinq.median(), inplace=True)


# In[22]:


##impute 'revol_util' with median too.
train_set.revol_util.fillna(train_set.revol_util.median(), inplace=True)
val_set.revol_util.fillna(train_set.revol_util.median(), inplace=True)


# In[23]:


## take a look at the dictionary with features kept
data_dictionary[data_dictionary.Feature.isin(train_set.columns)]


# In[26]:


## more features can be dropped as they only have one value.
for i in ['initial_list_status','policy_code','application_type','acc_now_delinq','delinq_amnt','pymnt_plan']:
    del train_set[i]
    del val_set[i]


# In[28]:


data_dictionary[data_dictionary.Feature.isin(train_set.columns)]


# In[25]:


## hist plot for all the columns
for i in [w for w in raw_reduced.columns if w not in ['emp_title','url','desc','zip_code','title']]:
        raw_reduced[i].hist()
        plt.title('distribution of '+i)
        plt.show()


# In[ ]:




