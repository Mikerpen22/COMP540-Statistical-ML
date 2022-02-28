from matplotlib.pyplot import axis
import numpy as np
from random import shuffle
import scipy.sparse


def softmax_loss_naive(theta, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - theta: d x K parameter matrix. Each column is a coefficient vector for class k
    - X: m x d array of data. Data are d-dimensional rows.
    - y: 1-dimensional array of length m with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to parameter matrix theta, an array of same size as theta
    """
    # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape
    K = theta.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in J and the gradient in grad. If you are not              #
    # careful here, it is easy to run into numeric instability. Don't forget    #
    # the regularization term!                                                  #
    #############################################################################
    # Loss calculation:
    pred = np.dot(X, theta)    # (m, K)
    # (m, 1) <- max will collapse the other dim, so add 1 to it
    max_pred_element = np.max(pred, axis=1).reshape((-1, 1))
    pred -= max_pred_element
    pred = np.exp(pred)
    pred_sum = np.sum(pred, axis=1).reshape((-1, 1))    # (m, 1)
    pred /= pred_sum    # (m, K)
    for i in range(m):
        for k in range(K):
            if(y[i] == k):
                J += np.log(pred[i, k])
    J /= (-m)
    J += reg * np.sum(np.power(theta, 2)) / 2 / m

    # Gradient calculation: -> grad shape (d, K)
    for k in range(K):
        for i in range(m):
            x_i = X[i, :]  # shape: (1, d)
            if y[i] == k:
                grad[:, k] += (x_i * (1 - pred[i, k]))  # shape: (1, d)
            else:
                grad[:, k] += (x_i * (0 - pred[i, k]))
        grad[:, k] /= (-m)
        grad[:, k] += reg * (theta[:, k])/m

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad


def softmax_loss_vectorized(theta, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape
    K = theta.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in J and the gradient in grad. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization term!                                                      #
    #############################################################################
    pred = np.dot(X, theta)    # (m, K)
    # (m, 1) <- max will collapse the other dim, so add 1 to it
    max_pred_element = np.max(pred, axis=1).reshape((-1, 1))
    pred -= max_pred_element
    pred = np.exp(pred)
    pred_sum = np.sum(pred, axis=1).reshape((-1, 1))    # (m, 1)
    pred /= pred_sum    # (m, K)

    # np.eye(number of classes)[vector containing the labels] -> cool!
    onehot = np.eye(K)[y]  # one-hot encoded label matrix -> (m, k)
    # A*B -> row by row multiplication
    J = -np.sum(onehot * np.log(pred))/m
    J += reg * np.sum(np.square(theta)) / 2.0 / m

    # grad shape: (d, K)
    # onehot - pred: (m, k)
    # X: (m, d)
    # theta: (d, K)

    grad = (np.matmul(X.T, (onehot-pred))/(-m)) + (reg * theta / m)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad
