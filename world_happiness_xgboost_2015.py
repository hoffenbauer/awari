#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
import xgboost as xgb 
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[32]:


dataset = loadtxt('/Users/Lucas/OneDrive/Desktop/2015_2.csv', delimiter=';', skiprows=1)


# In[33]:


dataset


# In[77]:


X = dataset[:,1:9]
Y = dataset[:,0]


# In[78]:


X[0]


# In[79]:


Y[0]


# In[93]:


train_X, test_X, train_Y, test_Y = train_test_split(X, Y, 
                      test_size = 0.3, random_state = 123) 


# In[98]:


xgb_r = xg.XGBRegressor(objective ='reg:squarederror', 
                  n_estimators = 10, seed = 123) 


# In[99]:


xgb_r.fit(train_X, train_Y) 


# In[90]:


y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[125]:


y_pred = model.predict(test_X)
pred = [round(value) for value in y_pred]


# In[112]:


rmse = np.sqrt(MSE(test_Y, pred)) 
print("RMSE : % f" %(rmse)) 


# In[126]:


accuracy = accuracy_score(test_Y, pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:




