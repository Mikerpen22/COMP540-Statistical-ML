import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import cholesky   # Need cholesky decomposition

## Sample from categorical distribution
def sample_categorical(cat, sample_size):
	# cat = [0.1, 0.1, 0.3, 0.3, 0.2]
	cat_cum = np.cumsum(cat)    # [0.1, 0.2, 0.5, 0.8 ,1.0]

	size = sample_size
	nums = np.random.rand(size) 

	condlist = [nums <= limit for limit in cat_cum]
	choicelist = [i for i in range(len(cat))]
	sample = np.select(condlist, choicelist)

	return sample


## Sample from univariate normal distribution (with Marsaglia polar method)
def sample_normal(mean=0, std=1, sample_size=10000):
	samples = []

	while(len(samples) < sample_size):
		U = np.random.uniform(-1, 1)
		V = np.random.uniform(-1, 1)
		S = U**2 + V**2
		if S < 1:
			k = np.sqrt(-2*np.log(S)/S)
			samples += [U*k, V*k]
	return samples


def sample_2dgaussian(mean=[1,1], cov= [[1,0.5],[0.5,1]]):
	A = cholesky(cov)
	dim = len(mean)
	z = np.array(sample_normal(0, 1, dim))
	x = mean + np.matmul(A, z.T)
	return x


def sample_mixture(means=[[1,1], [1,-1], [-1,1], [-1,-1]], cov=[[1,0],[0,1]]):
	res = np.zeros(2)
	for mean in means:
		res += sample_2dgaussian(mean, cov)
	res /= len(means)
	return res

def in_2dcircle(center_x, center_y, x, y):
	v_center = np.array([center_x, center_y])
	v_i = np.array([x, y])
	dist = np.linalg.norm(v_i-v_center)
	if dist > 1:
		return False
	return True




if __name__ == "__main__":

	## Runner for sampling categorical distribution
	# cat = [0.1, 0.1, 0.3, 0.3, 0.2]
	# c = sample_categorical(cat, 10000)
	# plt.hist(c)
	# plt.show()

	## Runner for sampling 1d normal distribution
	# a = sample_normal()
	# plt.hist(a)
	# plt.show()

	# Runner for sampling 2d multivariate guassian dist
	# Xs, Ys = [], []
	# for i in range(10000):
	# 	x,y = sample_2dgaussian()
	# 	Xs.append(x)
	# 	Ys.append(y)
	# plt.scatter(Xs, Ys)
	# plt.show()


	## Runner for sampling 2d mixture
	in_cnt, total_cnt = 0, 0
	X, Y = [], []
	for i in range(10000):
		total_cnt+=1
		# x,y = sample_2dgaussian()
		x, y = sample_mixture()
		X.append(x)
		Y.append(y)
		if in_2dcircle(0.1, 0.2, x, y):
			in_cnt += 1

	print(f"probability of samples in unit circle centering at (0.1,0.2): {in_cnt*1.0/total_cnt}")
	plt.scatter(X, Y)
	plt.show()







