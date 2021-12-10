#!/usr/bin/env python
# coding: utf-8

# ## Project part 2
# #### Author: Rahul Gore

# In[1]:


'''Imports'''
import scipy
import scipy.io
import numpy as np
from libsvm.svmutil import *
# import time


# In[2]:


train_data = scipy.io.loadmat('trainData.mat')
train_x1 = train_data['X1']
train_x1 = scipy.sparse.csr_matrix(train_x1)
train_x2 = train_data['X2']
train_x3 = train_data['X3']
train_y = train_data['Y'].reshape((4786,))


# In[3]:


test_data = scipy.io.loadmat('testData.mat')
test_x1 = test_data['X1']
test_x1 = scipy.sparse.csr_matrix(test_x1)
test_x2 = test_data['X2']
test_x3 = test_data['X3']
test_y = test_data['Y']


# ### Step 0

# In[4]:


h1 = svm_train(train_y, train_x1, '-c 10 -t 0 -b 1')
h2 = svm_train(train_y, train_x2, '-c 10 -t 0 -b 1')
h3 = svm_train(train_y, train_x3, '-c 10 -t 0 -b 1')


# In[5]:


label1, acc1, val1 = svm_predict(train_y, train_x1, h1)


# In[6]:


label2, acc2, val2 = svm_predict(train_y, train_x2, h2)


# In[7]:


label3, acc3, val3 = svm_predict(train_y, train_x3, h3)


# ### Step 1

# In[21]:


print(val3[0][0:10])


# In[ ]:


4700, 1225

