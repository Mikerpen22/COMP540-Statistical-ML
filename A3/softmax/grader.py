import graderUtil
import pandas as pd
import numpy as np
import os
import shutil
import random
import pickle as pkl
from random import randrange
import time

if __name__ == "__main__":
	grader = graderUtil.Grader()


	############################################################
	### Problem 6, Softmax regression (45 points)###
	############################################################
	softmax = grader.load('softmax')
	linear_classifier = grader.load("linear_classifier")
	with open('softmax_loss_history.pkl', 'rb') as f:
		loss_history_sol = pkl.load(f)

	LOSS_NAIVE = 0
	GRAD_NAIVE = 0
	###########################################################
	# Problem 6.1: Implementing the loss function for 
	# softmax regression (naive version) (5 points)
	###########################################################
	theta0 = np.random.randn(3073,10) * 0.0001
	LOSS0 = 0
	def check_6_1():
		for i in range(20):
			loss_naive, _ = softmax.softmax_loss_naive(theta0, X_train, y_train, 0.0)
			global LOSS0
			global LOSS_NAIVE
			LOSS0 += loss_naive/20.0
			LOSS_NAIVE = loss_naive
		grader.requireIsLessThan(0.05, np.abs(LOSS0 - 2.36))

	grader.addPart('6.1', check_6_1, 5)

	########################################################
	## Problem 6.2: Implementing the gradient of loss function 
	## for softmax regression (naive version) (5 points)
	########################################################

	
	
	def check_6_2():
		f = lambda th: softmax.softmax_loss_naive(th, X_train, y_train, 0.0)[0]
		_, grad_naive = softmax.softmax_loss_naive(theta0, X_train, y_train, 0.0)
		global GRAD_NAIVE
		GRAD_NAIVE = grad_naive
		x = theta0
		ix = tuple([randrange(m) for m in theta0.shape])
		h = 1e-5
		x[ix] += h # increment by h
		fxph = f(theta0) # evaluate f(x + h)
		x[ix] -= 2 * h # increment by h
		fxmh = f(theta0) # evaluate f(x - h)
		x[ix] += h # reset
		grad_numerical = (fxph - fxmh) / (2 * h)
		grad_analytic = grad_naive[ix]
		rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
		grader.requireIsLessThan(1e-6, rel_error)

	grader.addPart('6.2', check_6_2 , 5)

	#####################################################
	## Problem 6.4: Implementing the loss function for softmax regression (vectorized version) (10 points)
	####################################################
	def check_6_4_1():
		loss_vectorized, _ = softmax.softmax_loss_vectorized(theta0, X_train, y_train, 0.00001)
		grader.requireIsLessThan(1e-10, np.abs(LOSS_NAIVE - loss_vectorized))

	def check_6_4_2():
		naive_tic = time.time()
		_, _ = softmax.softmax_loss_naive(theta0, X_train, y_train, 0.0)
		naive_toc = time.time()
		time_for_naive = naive_toc - naive_tic
		vectorized_tic = time.time()
		_, _ = softmax.softmax_loss_vectorized(theta0, X_train, y_train, 0.00001)
		vectorized_toc = time.time()
		time_for_vectorized = vectorized_toc - vectorized_tic 
		grader.requireIsLessThan(time_for_naive, time_for_vectorized)
	grader.addPart('6.4.1', check_6_4_1, 5)
	grader.addPart('6.4.2', check_6_4_2, 5)

	#####################################################
	## Problem 6.5: Implementing the gradient of loss for softmax regression (vectorized version) (5 points)
	####################################################
	def check_6_5():
		_, grad_vectorized = softmax.softmax_loss_vectorized(theta0, X_train, y_train, 0.00001)
		grader.requireIsLessThan(1e-10, np.linalg.norm(GRAD_NAIVE - grad_vectorized, ord='fro'))
	grader.addPart('6.5', check_6_5, 5)

	#####################################################
	## Problem 6.6: Implementing mini-batch gradient descent (5 points))
	####################################################
	lc = linear_classifier.Softmax()
	lr = 1e-7
	reg = 5e4
	loss_history = None
	def check_6_6_1():
		global loss_history
		loss_history = lc.train(X_train, y_train, learning_rate = lr, reg = reg, num_iters = 4000, batch_size = 100, verbose = True)
		pred_val = lc.predict(X_test)
		pred_train = lc.predict(X_train)
		pred_ac = np.sum((pred_val == y_test).astype(int))/ len(y_test)
		grader.requireIsLessThan(0.2, np.abs(pred_ac - 0.278))

	def check_6_6_2():
		pred_val = lc.predict(X_test)
		pred_train = lc.predict(X_train)
		pred_ac = np.sum((pred_val == y_test).astype(int))/ len(y_test)
		train_ac = np.sum((pred_train == y_train).astype(int))/ len(y_train)
		grader.requireIsLessThan(0.2, np.abs(train_ac - 0.3357142857142857))

	def check_6_6_3():
		global loss_history_sol
		avr_sol = np.sum(np.abs(np.asarray(loss_history_sol[-10:]))) / 10
		avr = np.sum(np.abs(np.asarray(loss_history[-10:]))) / 10
		grader.requireIsLessThan(0.225, np.abs(avr - avr_sol))


	grader.addPart('6.6.1', check_6_6_1, 1)
	grader.addPart('6.6.2', check_6_6_2, 1)
	grader.addPart('6.6.3', check_6_6_3, 3)

	#################################################
	# Problem 6.7: Using a validation set to select regularization 
	# lambda and learning rate for gradient descent (5 points)
	#################################################
	grader.addManualPart('6.7', 5)

	############################################
	## Problem 6.8: Training a softmax classifier with 
	## the best hyperparameters (5 points)
	############################################
	grader.addManualPart('6.8', 5)

	############################################
	## Extra credit: Problem 6.10: Experimenting with 
	## other hyper parameters and optimization method (10 points)
	############################################
	grader.addManualPart('6.10', 10)


	##########################################
	# Extra credit: Problem 6.11. Building GDA 
	# classifiers for CIFAR10 (10 points)
	##########################################
	grader.addManualPart('6.11', 10)

	grader.grade()
