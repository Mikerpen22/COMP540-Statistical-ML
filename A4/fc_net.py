import numpy as np
from sklearn.preprocessing import scale

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of d, a hidden dimension of h, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a zero-mean Gaussian with stdev equal to      #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'theta1' and 'theta1_0' and second     #
        # layer weights and biases using the keys 'theta2' and 'theta2_0.          #
        # theta1 has shape (input_dim,hidden-dim), theta1_0 shape is (hidden_dim,) #
        # theta2 shape is (hidden_dim,num_classes), theta2_0 shape is (num_classes,)#
        ############################################################################
        # 4 lines of code expected
        self.params["theta1"] = np.random.normal(
            loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params["theta2"] = np.random.normal(
            loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params["theta1_0"] = np.zeros((hidden_dim,))
        self.params["theta2_0"] = np.zeros((num_classes,))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (m, d_1, ..., d_k)
        - y: Array of labels, of shape (m,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (m, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # Hint: unpack the weight parameters from self.params
        # then calculate output of two layer network using functions defined before
        # 3 lines of code expected
        out_1, cache_1 = affine_relu_forward(
            X, self.params["theta1"], self.params["theta1_0"])
        out_2, cache_2 = affine_forward(
            out_1, self.params["theta2"], self.params["theta2_0"])

        scores = out_2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # 4-8 lines of code expected
        loss, dx = softmax_loss(out_2, y)
        loss += 0.5 * self.reg * \
            (np.sum(np.power(self.params["theta1"], 2)) +
             np.sum(np.power(self.params["theta2"], 2)))

        # Backprop: last layer -> hidden layer
        dx, dtheta, dtheta0 = affine_backward(dx, cache_2)
        grads["theta2"] = dtheta + self.reg * self.params["theta2"]
        grads["theta2_0"] = dtheta0   # bias term doesn't need regularization

        # Backprop: hidden layer -> input layer
        dx, dtheta, dtheta0 = affine_relu_backward(dx, cache_1)
        grads["theta1"] = dtheta + self.reg * self.params["theta1"]
        grads["theta1_0"] = dtheta0
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout as an option. For a network with L layers,
    the architecture will be

    {affine - - relu - [dropout]} x (L - 1) - affine - softmax

    where  dropout is  optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 2 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in theta1 and theta1_0 for the second layer use theta2 and theta2_0, etc.#
        # Weights should beinitialized from a normal distribution with standard    #
        # deviation equal to weight_scale and biases should be initialized to zero.#
        #                                                                          #
        ############################################################################
        # about 4 lines of code
        for i in range(1, self.num_layers):
            dim_prev, dim_curr = 0, 0

            if i == 1:                    # input layer
                dim_prev = input_dim
                dim_curr = hidden_dims[i-1]
            elif i == self.num_layers-1:  # last layer
                dim_prev = hidden_dims[i-2]
                dim_curr = num_classes
            else:
                dim_prev = hidden_dims[i-2]
                dim_curr = hidden_dims[i-1]

            self.params[f"theta{i}"] = np.random.normal(
                loc=0.0, scale=weight_scale, size=(dim_prev, dim_curr)).astype(dtype)
            self.params[f"theta{i}_0"] = np.zeros((dim_curr,)).astype(dtype)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # Cast all parameters to the correct datatype

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for  dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        ############################################################################
        outs, caches = {}, {}     # Store results for every step of forward pass
        for i in range(1, self.num_layers):
            if i == self.num_layers-1:  # Last layer
                if self.use_dropout:
                    outs[f"theta{i}"], caches[f"theta{i}"] = affine_forward(
                        outs[f"dropout{i-1}"], self.params[f"theta{i}"], self.params[f"theta{i}_0"])
                else:
                    outs[f"theta{i}"], caches[f"theta{i}"] = affine_forward(
                        outs[f"theta{i-1}"], self.params[f"theta{i}"], self.params[f"theta{i}_0"])
            else:
                if i == 1:
                    outs[f"theta{i}"], caches[f"theta{i}"] = affine_relu_forward(
                        X, self.params[f"theta{i}"], self.params[f"theta{i}_0"])
                    if self.use_dropout:
                        outs[f"dropout{i}"], caches[f"dropout{i}"] = dropout_forward(
                            outs[f"theta{i}"], self.dropout_param)
                else:
                    if self.use_dropout:
                        outs[f"theta{i}"], caches[f"theta{i}"] = affine_relu_forward(
                            outs[f"dropout{i-1}"], self.params[f"theta{i}"], self.params[f"theta{i}_0"])
                        outs[f"dropout{i}"], caches[f"dropout{i}"] = dropout_forward(
                            outs[f"theta{i}"], self.dropout_param)
                    else:
                        outs[f"theta{i}"], caches[f"theta{i}"] = affine_relu_forward(
                            outs[f"theta{i-1}"], self.params[f"theta{i}"], self.params[f"theta{i}_0"])

        scores = outs[f"theta{self.num_layers-1}"]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * \
            sum([np.sum(np.power(theta[1], 2))
                for theta in self.params.items()])

        # Back prop
        for i in range(self.num_layers-1, 0, -1):
            if self.use_dropout and i != self.num_layers-1:
                dx = dropout_backward(dx, caches[f"dropout{i}"])

            if i == self.num_layers-1:
                dx, dtheta, dtheta0 = affine_backward(dx, caches[f"theta{i}"])
            else:
                dx, dtheta, dtheta0 = affine_relu_backward(
                    dx, caches[f"theta{i}"])

            grads[f"theta{i}"] = dtheta + self.reg * self.params[f"theta{i}"]
            grads[f"theta{i}_0"] = dtheta0

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
