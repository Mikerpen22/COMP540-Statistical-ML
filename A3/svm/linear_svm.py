import numpy as np


##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension d, there are K classes, and we operate on minibatches
    of m examples.

    Inputs:
    - theta: A numpy array of shape d x K containing parameters.
    - X: A numpy array of shape m x d containing a minibatch of data.
    - y: A numpy array of shape (m,) containing training labels; y[i] = k means
            that X[i] has label k, where 0 <= k < K.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss J as single float
    - gradient with respect to weights theta; an array of same shape as theta
    """

    K = theta.shape[1]  # number of classes
    m = X.shape[0]     # number of examples

    J = 0.0
    # initialize the gradient as zero, shape = (d, K)
    dtheta = np.zeros(theta.shape)
    delta = 1.0

    #############################################################################
    # TODO:                                                                     #
    # Compute the loss function and store it in J.                              #
    # Do not forget the regularization term!                                    #
    # code above to compute the gradient.                                       #
    # 8-10 lines of code expected                                               #
    #############################################################################
    pred = np.dot(X, theta)  # (m, k)

    for i in range(m):
        pred_i = pred[i]  # (1, k)
        margins = pred_i - pred[i, y[i]] + delta
        margins[y[i]] = 0  # loss doesn't consider j == y[i] -> let it be 0!
        for idx, margin in enumerate(margins):  # len(margins) = K
            if idx == y[i]:  # loss doesn't consider this case -> so grad won't have this
                continue
            if margin > 0:  # margin < 0 -> will be clipped -> no derivative
                dtheta[:, idx] += X[i]		# X[i]'s shape: (1, d)
                dtheta[:, y[i]] -= X[i]

        margins = np.clip(margins, 0, None)
        J += np.sum(margins)

        # Update the grad

    J /= m
    J += reg * np.sum(np.square(theta)) / 2.0
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dtheta.            #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    dtheta = 1.0*dtheta/m
    dtheta += reg * theta

    return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    J = 0.0
    dtheta = np.zeros(theta.shape)  # initialize the gradient as zero
    delta = 1.0
    m = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in variable J.                                                     #
    # 8-10 lines of code                                                        #
    #############################################################################
    pred = np.dot(X, theta)  # (m, k)
    pred_correct = pred[np.arange(m), y].reshape((-1, 1))  # (m, 1)
    margins = pred - pred_correct + delta  # (m, k)
    margins = np.clip(margins, 0, None)
    margins[np.arange(m), y] = 0

    J = 1.0*np.sum(margins)/m + reg * np.sum(np.square(theta)) / 2.0

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dtheta.                                       #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # (m, k), also explicitly cast bool to int or it can't be used to subtract
    margins_mask = (margins > 0).astype(float)

    # For every row in margins:
    # we will subtract X[i] for [margin>0] times
    # add X[i] for [margin > 0] times
    # shape: (m, k)
    margins_mask[np.arange(m), y] -= np.sum(margins_mask, axis=1)
    dtheta = np.dot(X.T, margins_mask) / m + reg * theta

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return J, dtheta
