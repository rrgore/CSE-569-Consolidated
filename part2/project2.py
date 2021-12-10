#!/usr/bin/env python
# coding: utf-8

# ## Project part 2
# #### Author: Rahul Gore

'''Imports'''
import scipy
import scipy.io
import numpy as np
from libsvm.svmutil import *


train_data = scipy.io.loadmat('trainData.mat')
train_x1 = np.array(train_data['X1'])
train_x2 = np.array(train_data['X2'])
train_x3 = np.array(train_data['X3'])
train_y = train_data['Y'].flatten()


test_data = scipy.io.loadmat('testData.mat')
test_x1 = np.array(test_data['X1'])
test_x2 = np.array(test_data['X2'])
test_x3 = np.array(test_data['X3'])
test_y = test_data['Y'].flatten()


# ### Step 0 (1)
# Classification models
class_h1 = svm_train(train_y, train_x1, '-c 10 -t 0 -q')
class_h2 = svm_train(train_y, train_x2, '-c 10 -t 0 -q')
class_h3 = svm_train(train_y, train_x3, '-c 10 -t 0 -q')


label1, acc1, val1 = svm_predict(test_y, test_x1, class_h1)
label2, acc2, val2 = svm_predict(test_y, test_x2, class_h2)
label3, acc3, val3 = svm_predict(test_y, test_x3, class_h3)


# ### Step 0 (2)
# Probability models
prob_h1 = svm_train(train_y, train_x1, '-c 10 -t 0 -b 1 -q')
prob_h2 = svm_train(train_y, train_x2, '-c 10 -t 0 -b 1 -q')
prob_h3 = svm_train(train_y, train_x3, '-c 10 -t 0 -b 1 -q')


label1, acc1, val1 = svm_predict(test_y, test_x1, prob_h1, "-b 1")
label2, acc2, val2 = svm_predict(test_y, test_x2, prob_h2, "-b 1")
label3, acc3, val3 = svm_predict(test_y, test_x3, prob_h3, "-b 1")


# ### Step 1
vals = [val1, val2, val3]
vals = np.array(vals)
stack_val = np.stack(vals, axis=-1)
fusion = np.mean(stack_val, axis=2)


fusion_pred = np.argmax(fusion, axis=1)
fusion_pred += 1


match = []
for i in range(1883):
    if fusion_pred[i] == test_y[i]:
        match.append(1)
    else:
        match.append(0)
print("Accuracy: ", sum(match)/18.83)


# ### Step 2
# Combined features
train_features = np.stack((train_x1, train_x2, train_x3), axis=1)
train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1]*train_features.shape[2]))


comb_h = svm_train(train_y, train_features, '-c 10 -t 0 -q')


test_features = np.stack((test_x1, test_x2, test_x3), axis=1)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1]*test_features.shape[2]))


label, acc, val = svm_predict(test_y, test_features, comb_h1)