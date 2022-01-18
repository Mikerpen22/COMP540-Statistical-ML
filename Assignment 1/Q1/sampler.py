import numpy as np
import matplotlib.pyplot as plt


def categorical_sample(class_prob, n_samples):
    '''
    Generate categorical samples

    Arguments: 
    class_prob: the weights of the different components of work required for this class
    n_samples: the number of samples

    Return:
    dataset_cate: simulated categorical samples
    '''
    n_classes = len(class_prob)
    class_cumsum = np.cumsum(class_prob)
    dataset = np.random.uniform(size=n_samples)
    lower_limits = [dataset <= limit for limit in class_cumsum]
    class_matrix = [i * np.ones(n_samples) for i in range(n_classes)]
    dataset_cate = np.select(condlist=lower_limits, choicelist=class_matrix)
    dataset_cate = [int(i) for i in dataset_cate]
    return dataset_cate


def normal_sample(mean, variance, n_samples):
    '''
    Generate normal samples

    Arguments: 
    mean: mean of the normal distribution
    variance: variance of the normal distribution
    n_samples: the number of samples

    Return:
    Y: simulated normal samples
    '''
    U = np.random.uniform(size=n_samples*2)
    U_1 = U[:n_samples]
    U_2 = U[n_samples:]

    X_1 = np.sqrt(-2 * np.log(U_1)) * np.cos(2 * np.pi * U_2)
    X = X_1
    Y = mean + np.sqrt(variance) * X
    return Y


def multinormal_sample(mean, cov, n_samples):
    '''
    Generate multivariate Gaussian (2-D) samples

    Arguments: 
    mean: means of the multivariate Gaussian distribution
    cov: covariance matrix
    n_samples: the number of samples

    Return:
    Y: simulated 2-D Gaussian samples
    '''
    U = np.random.uniform(size=n_samples*2)
    U_1 = U[:n_samples]
    U_2 = U[n_samples:]

    X_1 = np.sqrt(-2 * np.log(U_1)) * np.cos(2 * np.pi * U_2)
    X_2 = np.sqrt(-2 * np.log(U_1)) * np.sin(2 * np.pi * U_2)
    X = np.concatenate((X_1, X_2), axis=0).reshape((n_samples, 2))

    L = np.linalg.cholesky(cov)
    Y = mean + X @ L.T

    return Y


def mixture_sample(centers, cov, coefficients, n_samples):
    '''
    Generate mixture Gaussian (2-D) samples

    Arguments: 
    centers: means of the multivariate Gaussian distributions
    cov: covariance matrix
    coefficients: weights for each Gaussian
    n_samples: the number of samples

    Return:
    Y: simulated 2-D Gaussian samples
    '''
    num_distr = len(centers)
    num_dim = len(centers[0])

    data = np.zeros((n_samples, num_distr, num_dim))

    for idx, center in enumerate(centers):
        data[:, idx, :] = multinormal_sample(center, cov, n_samples)

    random_idx = np.random.choice(
        np.arange(num_distr), size=(n_samples,), p=coefficients)

    sample = data[np.arange(n_samples), random_idx, :]

    return sample


if __name__ == "__main__":

    '''Plot for categorical samples'''
    plt.style.use('fivethirtyeight')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    cate_samples = categorical_sample([0.1, 0.1, 0.3, 0.3, 0.2], 10000)
    ax1.hist(cate_samples, range=(-0.5, 4.5), align='left')
    ax1.set_xlabel('Categorical data')
    ax1.set_ylabel('Count')

    '''Plot for normal samples'''
    normal_samples = normal_sample(1, 1, 10000)
    ax2.hist(normal_samples, bins=100)
    ax2.set_xlabel('Normal data')
    ax2.set_ylabel('Count')

    '''Plot for multinormal samples'''
    multinormal_samples = multinormal_sample(
        [1, 1], [[1, 0.5], [0.5, 1]], 10000)
    ax3.scatter(multinormal_samples[:, 0], multinormal_samples[:, 1])
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')

    '''Plot for Mixture model samples'''
    mixture_samples = mixture_sample(centers=[(1, 1), (1, -1), (-1, 1), (-1, -1)],
                                     cov=np.identity(2), coefficients=[0.25, 0.25, 0.25, 0.25], n_samples=100000)
    ax4.scatter(mixture_samples[:, 0], mixture_samples[:, 1])
    ax4.set_xlabel('X1')
    ax4.set_ylabel('X2')
    plt.tight_layout()
    plt.show()

    n_in = 0
    for item in mixture_samples:
        if (item[0]-0.1) ** 2 + (item[1]-0.2) ** 2 <= 1:
            n_in += 1

    print('probability of sample lies within the unit circle centered at (0.1,0.2) is: ' + str(n_in/100000))
