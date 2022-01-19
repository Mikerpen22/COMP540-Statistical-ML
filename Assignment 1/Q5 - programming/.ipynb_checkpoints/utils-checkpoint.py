from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
import plot_utils


#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values

    mu = X.mean(axis=0)
    sigma = X.mean(axis=0)
    X_norm = (X-mu)/sigma
  
    ########################################################################
    return X_norm, mu, sigma


#############################################################################
#  Plot the learning curve for training data (X,y) and validation set       #
# (Xval,yval) and regularization lambda reg.                                #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
    
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 7 lines of code expected                                                #
    ###########################################################################
    reglinear_reg = RegularizedLinearReg_SquaredLoss()
    for i in range(1,num_examples):
        X_train_sub = X[:i]
        y_train_sub = y[:i]
        theta_opt = reglinear_reg.train(X_train_sub,y_train_sub,reg,num_iters=1000)        
        J_train_sub = reglinear_reg.loss(theta_opt, X_train_sub, y_train_sub, 0)
        J_val = reglinear_reg.loss(theta_opt, Xval, yval, 0)
        
        error_train[i], error_val[i] = J_train_sub, J_val
        
    ###########################################################################

    return error_train, error_val

#############################################################################
#  Plot the validation curve for training data (X,y) and validation set     #
# (Xval,yval)                                                               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#                                                                           #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def validation_curve(X,y,Xval,yval):
  
    reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = np.zeros((len(reg_vec),))
    error_val = np.zeros((len(reg_vec),))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 5 lines of code expected                                                #
    ###########################################################################
    reglinear_reg = RegularizedLinearReg_SquaredLoss()
    for i,reg in enumerate(reg_vec):
        theta_opt = reglinear_reg.train(X,y,reg,num_iters=1000)        
        J_train = reglinear_reg.loss(theta_opt, X, y, 0)
        J_val = reglinear_reg.loss(theta_opt, Xval, yval, 0)
        
        error_train[i], error_val[i] = J_train, J_val
    
    return reg_vec, error_train, error_val

import random

#############################################################################
#  Plot the averaged learning curve for training data (X,y) and             #
#  validation set  (Xval,yval) and regularization lambda reg.               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def averaged_learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
       
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 10-12 lines of code expected                                            #
    ###########################################################################
    avg_size = 50
    reglinear_reg = RegularizedLinearReg_SquaredLoss()
    for i in range(1, num_examples):
        J_train_sub, J_val = 0, 0
        for j in range(avg_size):
            rand_ints = np.random.choice([k for k in range(len(X))], (i,))

            X_train_sub = X[rand_ints]
            y_train_sub = y[rand_ints]
            Xval_sub = Xval[rand_ints]
            yval_sub = yval[rand_ints]
            theta_opt = reglinear_reg.train(X_train_sub,y_train_sub,reg,num_iters=1000)     
            J_train_sub += reglinear_reg.loss(theta_opt, X_train_sub, y_train_sub, 0)
            J_val += reglinear_reg.loss(theta_opt, Xval, yval, 0)    

        error_train[i], error_val[i] = J_train_sub/avg_size, J_val/avg_size


    ###########################################################################
    return error_train, error_val


#############################################################################
# Utility functions
#############################################################################
    
def load_mat(fname):
  d = scipy.io.loadmat(fname)  # load matlab file
  X = d['X']
  y = d['y']
  Xval = d['Xval']
  yval = d['yval']
  Xtest = d['Xtest']
  ytest = d['ytest']

  # need reshaping!

  X = np.reshape(X,(len(X),))
  y = np.reshape(y,(len(y),))
  Xtest = np.reshape(Xtest,(len(Xtest),))
  ytest = np.reshape(ytest,(len(ytest),))
  Xval = np.reshape(Xval,(len(Xval),))
  yval = np.reshape(yval,(len(yval),))

  return X, y, Xtest, ytest, Xval, yval









