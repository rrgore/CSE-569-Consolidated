import scipy
import scipy.io
import numpy as np
from libsvm.svmutil import *
import time

train_data = scipy.io.loadmat('trainData.mat')
train_x1 = train_data['X1']
train_x1 = scipy.sparse.csr_matrix(train_x1)
train_x2 = train_data['X2']
train_x3 = train_data['X3']
train_y = train_data['Y'].reshape((4786,))
start_time = time.time()

h1 = svm_train(train_y, train_x1, '-c 10 -t 0 -b 1')
p_label, p_acc, p_val = svm_predict(train_y, train_x1, h1)
print("--- %s seconds ---" % (time.time() - start_time))