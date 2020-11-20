#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
from collections import defaultdict
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn import svm


# In[2]:


X_test = pd.read_csv('Xtest.txt', sep=' ')
Y_test = pd.read_csv('Ytest.txt', sep=',')
Y_train = pd.read_csv('Ytrain.txt', sep=' ')
X_train = pd.read_csv('Xtrain.txt', sep=' ')
X_test = X_test.fillna(0)
X_train = X_train.fillna(0)
Y_train = Y_train.fillna(0)
Y_train=Y_train.replace('?', 0)
X_train=X_train.replace('?', 0)
X_test = X_test.replace('?', 0)


# In[3]:


Y_train = Y_train.drop('Id', 1)
X_train = X_train.drop('Id', 1)
X_test = X_test.drop('Id', 1)


# In[4]:


a = DotProduct() + WhiteKernel()


# In[5]:



d = {}
Y_train['Z01']
for i in range(1, 15, 1):

    d["m" + str(i)] = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features='sqrt',
                          n_jobs=-1)
    col = 'Z' + '0' + str(i) if i<10 else 'Z'+ str(i)
    print(col)
    d["m"+str(i)].fit(X_train, Y_train[col])


# In[ ]:


for i in range(1, 15, 1):
    col = 'Z' + '0' + str(i) if i<10 else 'Z'+ str(i)
    print(col)
    Y_test[col] = d["m"+str(i)].predict(X_test, return_std=True)


# In[ ]:


Y_test['index'] = Y_test['Id'].str.split(":", n = 1, expand = True)[1]
indexArr = []
for index, row in Y_test.iterrows(): 
#     print(row[row["index"]])
    indexArr.append(row[row["index"]])
tmp =np.asarray(indexArr)
Y_test['Value'] = tmp
Y_test['Id'] = Y_test['Id'].str.split(":", n = 1, expand = True)[0]


# In[ ]:


Y_test = Y_test[['Id','Value']]
Y_test.to_csv('result.csv',index=False)

