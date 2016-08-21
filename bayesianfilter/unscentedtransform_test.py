# coding:utf-8
import numpy as np

from scipy.stats import multivariate_normal


class UT(object):
    def __init__(self):
        self.N = 2
        alpha, beta, kappa = 1, 0, 1

        self.scaling_param = alpha ** 2 * (self.N + kappa) - self.N

        self.wm, self.wc = self.get_weights(alpha, beta)

    def get_weights(self, alpha, beta):
        """Computes weights for unscented transform.

        returns the mean weight and covariance weight in a tuple.

        Parameters
        ----------
        alpha : float
            scaling parameter.

        beta : float
            scaling parameter.

        Returns
        -------
        wm : ndarray, shape = (2 * state_dim + 1)
            weight for mean.
        wc : ndarray, shape = (2 * state_dim + 1)
            weigth for covariance.
        """
        wm = np.empty(2 * self.N + 1)
        wc = np.empty(2 * self.N + 1)
        wm[0] = self.scaling_param / (self.N + self.scaling_param)
        wm[1:] = 1 / (2 * (self.N + self.scaling_param))

        wc[0] = self.scaling_param / (self.N + self.scaling_param) + (1 - alpha ** 2 + beta)
        wc[1:] = 1 / (2 * (self.N + self.scaling_param))

        return wm, wc

    def get_sigma_points(self, mu, P):
        # calc sigma points
        sigma_points = [mu]
        root_P = np.linalg.cholesky(P)

        for i in range(self.N):
            sigma_points.append(mu + (self.N + self.scaling_param) ** 0.5 * root_P[:, i])

        for i in range(self.N):
            sigma_points.append(mu - (self.N + self.scaling_param) ** 0.5 * root_P[:, i])
        sigma_points = np.array(sigma_points)

        return sigma_points

    def unscented_transform(self, mean, covariance, propagate_function):
        """Computes unscented transform of a set of sigma points and weights

        returns the mean and covariance in a tuple.

        Parameters
        ----------
        mean : ndarray, shape = (state_dim)

        covariance : ndarray, shape = (state_dim, state_dim)

        propagate_function : function
            Function that computes the sigmapoints

        Returns
        -------
        transformed_mean : ndarray, shape = (dimension of propagate function)
            Mean of the sigma points after passing through the transform.

        transformed_covariance : ndarray, shape =(dimension of propagate function,dimension of propagate function)
            covariance of the sigma points after passing throgh the transform.

        """
        sigma_points = self.get_sigma_points(mean, covariance)
        # propagete
        propagated_sigma_points = []
        for i in range(self.N * 2 + 1):
            propagated_sigma_points.append(propagate_function(sigma_points[i]))
        propagated_sigma_points = np.array(propagated_sigma_points)

        transformed_mean = np.zeros(propagated_sigma_points.shape[1])
        transformed_covariance = np.zeros([propagated_sigma_points.shape[1], propagated_sigma_points.shape[1]])

        for i in range(0, 2 * self.N + 1):
            transformed_mean += self.wm[i] * propagated_sigma_points[i, :]

        for i in range(0, 2 * self.N + 1):
            transformed_covariance += self.wc[i] * np.outer(propagated_sigma_points[i] - transformed_mean,
                                                            propagated_sigma_points[i] - transformed_mean)

        return transformed_mean, transformed_covariance


def g(X):
    return np.r_[X[0] ** 3, X[1] ** 3] / 100


if __name__ == "__main__":
    ut = UT()

    mu = np.array([3, 2])
    P = np.array([[4, 1], [1, 1]])

    X = np.random.multivariate_normal(mu, P, size=1000)
    Y = []
    for x in X:
        Y.append(g(x))

    Y = np.array(Y)
    print(Y.shape)
    transformed_mu, transformed_cov = ut.unscented_transform(mu, P, g)

    x, y = np.mgrid[np.min(Y[:, 0]):np.max(Y[:, 0]):1, np.min(Y[:, 1]):np.max(Y[:, 1]):1]

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    print(g(mu), transformed_mu)
    rv = multivariate_normal(transformed_mu, transformed_cov)
    import matplotlib.pyplot as plt

    plt.contour(x, y, rv.pdf(pos))

    plt.scatter(transformed_mu[0], transformed_mu[1], c="g")
    plt.scatter(Y[:, 0], Y[:, 1], c="g")
    plt.show()
