import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def plot_normal_cdf():
    x = np.arange(-4,4.01,0.01)
    y = stats.norm.pdf(x, loc = 0, scale = 1)

    plt.plot(x,y)
    plt.show()


def plot_normal_copula(cor, d, points):
    A = np.linalg.cholesky(cor)
    samples = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d), size = points)
    transformed = samples @ A.T
    return stats.norm.cdf(transformed)

copula = plot_normal_copula(
    np.array([[1.0, 0.6],
              [0.6, 1.0]]), 
              2,
              1000)
plt.scatter(copula[:,0], copula[:,1], alpha = 0.5)
plt.show()