#!/usr/bin/env python
# coding: utf-8

# # FSL Project Part 1
# Feature Extraction, Density Estimation and Bayesian Classification
# Author - Rahul Gore

''' Imports '''
import scipy.io
from scipy.spatial import distance
import numpy as np
from numpy.linalg import inv, det
import math


def getNormalizedFeatures(data_arr):
    ''' Return array of normalized vectors '''
    n_imgs = data_arr.shape[0]
    # print(n_imgs)
    m_arr = np.zeros(n_imgs)
    s_arr = np.zeros(n_imgs)

    for i in range(n_imgs):
        m_arr[i] = np.mean(data_arr[i])
        s_arr[i] = np.std(data_arr[i])
        
    M1 = np.mean(m_arr)
    M2 = np.mean(s_arr)
    S1 = np.std(m_arr)
    S2 = np.std(s_arr)
    
    y = np.zeros((n_imgs, 2))

    for i in range(n_imgs):
        y[i] = [(m_arr[i]-M1)/S1, (s_arr[i]-M2)/S2]
    
    return np.transpose(y)


def discrim(x, mean, cov, prior):
    ''' Discriminant for given likelihood and priors '''
    maha = distance.mahalanobis(x, mean, inv(cov))
    expFac = math.exp(-0.5 * (maha**2))
    const = 2 * math.pi * math.sqrt(det(cov))
    like = expFac * prior / const
    return like


def outputErrorProb(y_data, mu_3, sigma_3, omega_3,
                    mu_7, sigma_7, omega_7, type):
    # output = np.zeros(n_train)
    size = y_data.shape[0]
    error = np.zeros(size)
    for i in range(size):
        l3 = discrim(y_data[i], mu_3, sigma_3, omega_3)
        l7 = discrim(y_data[i], mu_7, sigma_7, omega_7)
        if l3 > l7:
            # train_case1_output[i] = 3
            error[i] = l7
        else:
            # train_case1_output[i] = 7
            error[i] = l3

    print("{0}: Error probability is {1}".format(type, np.mean(error)))


def driver():
    ''' Import training dataset '''
    train_data = scipy.io.loadmat('train_data.mat')
    train_data_arr = np.array(train_data['data'])
    label_arr = np.array(train_data['label'][0])
    n_train = train_data_arr.shape[0]

    ''' Import test dataset '''
    test_data = scipy.io.loadmat('test_data.mat')
    test_data_arr = np.array(test_data['data'])
    test_label_arr = np.array(test_data['label'][0])
    n_test = test_data_arr.shape[0]

    train_y = getNormalizedFeatures(train_data_arr)

    test_y = getNormalizedFeatures(test_data_arr)

    ''' Split features into 3 and 7 classes '''
    feature_3 = []
    feature_7 = []

    train_y = np.transpose(train_y)
    test_y = np.transpose(test_y)
    for i in range(n_train):
        if label_arr[i] == 3:
            feature_3.append(train_y[i])
        else:
            feature_7.append(train_y[i])

    feature_3 = np.array(feature_3)
    feature_7 = np.array(feature_7)

    ''' Calculate mean '''
    mu_3 = np.mean(feature_3, 0)
    mu_7 = np.mean(feature_7, 0)

    ''' Calculate variance '''
    len_3 = len(feature_3)
    sum_3 = 0
    for i in range(len_3):
        mat = (feature_3[i] - mu_3).reshape(2,1)
        trans = mat.reshape(1,2)
        sum_3 += mat @ trans
    sigma_3 = (1/len_3)*sum_3

    len_7 = len(feature_7)
    sum_7 = 0
    for i in range(len_7):
        mat = (feature_7[i] - mu_7).reshape(2,1)
        trans = mat.reshape(1,2)
        sum_7 += mat @ trans
    sigma_7 = (1/len_7)*sum_7

    print("Parameters for 3: mean={0}, covariance={1}".format(mu_3, sigma_3))
    print("Parameters for 7: mean={0}, covariance={1}".format(mu_7, sigma_7))

    outputErrorProb(train_y, mu_3, sigma_3, 0.5, 
                    mu_7, sigma_7, 0.5, "Training Data Case 1")
    outputErrorProb(train_y, mu_3, sigma_3, 0.3, 
                    mu_7, sigma_7, 0.7, "Training Data Case 2")
    outputErrorProb(test_y, mu_3, sigma_3, 0.5, 
                    mu_7, sigma_7, 0.5, "Test Data Case 1")
    outputErrorProb(test_y, mu_3, sigma_3, 0.3, 
                    mu_7, sigma_7, 0.7, "Test Data Case 2")


if __name__ == "__main__":
    driver()