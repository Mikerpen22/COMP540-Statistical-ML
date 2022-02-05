import numpy as np
from numpy.linalg import inv
import math


def sigmoid(x):			
	return 1/(1+np.exp(-x))

def hessian(X, theta):
	m = X.shape[0]
	# h_i= sigmoid(theta.T * x_i)
	diag = []
	for i in range(m):
		h_i = sigmoid(np.dot(theta.T, X[i])[0])
		print(h_i)
		diag.append(h_i)
	S = np.zeros((m,m), float)
	print(S)
	for i in range(m):
		S[i][i] = diag[i]

	H = np.matmul(X.T ,np.matmul(S,X))
	return H

def obj_func(X, y, theta, alpha):
	m = X.shape[0]
	d = X.shape[1]
	accum = np.zeros((3,))

	for i in range(m):
		mu_i = sigmoid(np.dot(theta.T, X[i])[0])
		y_i = y[i][0]
		print(X[i].shape)
		accum += (mu_i - y_i) * X[i]

	reg = (alpha/m)*sum(theta)

	return accum/m + reg


def nm_optim(X, y, theta, alpha):
	b = theta

	delta = 100
	itr = 0
	while itr < 10:
		b = b - np.dot(inv(hessian(X,theta)), obj_func(X, y, theta, alpha))
		print(f"iteration {itr}: theta = {b}")
		itr+=1
	print(b)

X = np.array([[1,0,3], [1,1,3], [1,0,1], [1,1,1]]).reshape(4,3)
y = np.array([1,1,0,0]).reshape(4,1)
theta = np.array([0, -2, 1]).reshape(3, 1)
alpha = 0.07



nm_optim(X,y,theta, alpha)

