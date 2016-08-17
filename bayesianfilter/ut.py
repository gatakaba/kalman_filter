# coding:utf-8
import numpy as np

import matplotlib.pyplot as plt

np.random.seed(0)


def g(X):
    return np.c_[X[:, 0] ** 3 / 100, X[:, 1] ** 3 / 100]


def get_sigmapoints(mu, P):
    sigma_points = [mu]
    for i in range(n):
        sigma_points.append(mu + (n + scaling_param) ** 0.5 * root_P[:, i])

    for i in range(n):
        sigma_points.append(mu - (n + scaling_param) ** 0.5 * root_P[:, i])


mu = np.array([3, 2])
P = np.array([[4, 1], [1, 1]])

X = np.random.multivariate_normal(mu, P, size=1000)
Y = g(X)

alpha = 1
beta = 0

kappa = 1
n = 2
scaling_param = alpha ** 2 * (n + kappa) - n
root_P = np.linalg.cholesky(P)

sigma_points = [mu]

for i in range(n):
    sigma_points.append(mu + (n + scaling_param) ** 0.5 * root_P[:, i])

for i in range(n):
    sigma_points.append(mu - (n + scaling_param) ** 0.5 * root_P[:, i])

sigma_points = np.array(sigma_points)
transformed_sigma_points = g(sigma_points)

mean0_weight = scaling_param / (n + scaling_param)
mean_weight = 1 / (2 * (n + scaling_param))
covariance0_weight = scaling_param / (n + scaling_param) + (1 - alpha ** 2 + beta)
covariance_weight = 1 / (2 * (n + scaling_param))

transformed_mu = mean0_weight * transformed_sigma_points[0, :]

for i in range(1, 2 * n + 1):
    transformed_mu += mean_weight * transformed_sigma_points[i, :]

transformed_cov = covariance0_weight * np.outer(transformed_sigma_points[0] - transformed_mu,
                                                transformed_sigma_points[0] - transformed_mu)

for i in range(1, 2 * n + 1):
    transformed_cov += covariance_weight * np.outer(transformed_sigma_points[i] - transformed_mu,
                                                    transformed_sigma_points[i] - transformed_mu)
print(transformed_cov)

plt.subplot(211)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(sigma_points[:, 0], sigma_points[:, 1], c="r")

plt.subplot(212)

from scipy.stats import multivariate_normal

x, y = np.mgrid[np.min(Y[:, 0]):np.max(Y[:, 0]):.01, np.min(Y[:, 1]):np.max(Y[:, 1]):.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
rv = multivariate_normal(transformed_mu, transformed_cov)
plt.contour(x, y, rv.pdf(pos))

plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5)
plt.scatter(transformed_sigma_points[:, 0], transformed_sigma_points[:, 1], c="r")
plt.scatter(transformed_mu[0], transformed_mu[1], c="g")
plt.show()
