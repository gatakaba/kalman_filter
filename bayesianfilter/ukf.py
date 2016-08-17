# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

"""
========================================
Inference for Dynamic NonLinear Systems
========================================
# fatures
    * estimate current state

"""


class UnscentedKalmanFilter(object):
    """ UnscentedKalmanFilter in python
        * p(z_{n+1}|z_{n}) = N(z_{n+1}|f(z_{n}),q_{k-1})
        * p(x_{n}|z_{n}) = N(x_{n}|h(z_{n}),r_{k})
    """

    def __init__(self, state_dim, observation_dim, transition_function, observation_function, process_noise,
                 observation_noise, initial_mean=None, initial_covariance=None):
        self.N = state_dim
        self.M = observation_dim

        self.f = transition_function
        self.h = observation_function

        self.Q = process_noise
        self.R = observation_noise

        self.mu = np.random.multivariate_normal(np.zeros(self.N), self.P)
        self.P = np.eye(self.N) * 10 ** 4

        alpha, beta, kappa = 1, 0, 1

        self.scaling_param = self.get_scaling_param(alpha, kappa)

        self.wm, self.wc = self.get_weights()
        self.check_parameter_size()

    def check_parameter_size(self):
        pass

    def get_scaling_param(self, alpha, kappa):
        return alpha ** 2 * (self.N + kappa) - self.N

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

    def unscented_transform(self, mu, P, propagete_function):
        """Computes unscented transform of a set of sigma points and weights

        returns the mean and covariance in a tuple.

        Parameters
        ----------
        mu : ndarray, shape = (state_dim)

        P : ndarray, shape = (state_dim, state_dim)

        propagete_function : function
            Function that computes the sigmapoints

        Returns
        -------
        transformed_mu : ndarray, shape = (state_dim)
            Mean of the sigma points after passing through the transform.

        transformed_P : ndarray, shape =(state_dim, state_dim)
            covariance of the sigma points after passing throgh the transform.

        """

        # calc sigma points
        sigma_points = [mu]
        root_P = np.linalg.cholesky(P)
        for i in range(self.N):
            sigma_points.append(mu + (self.N + self.scaling_param) ** 0.5 * root_P[:, i])

        for i in range(self.N):
            sigma_points.append(mu - (self.N + self.scaling_param) ** 0.5 * root_P[:, i])
        sigma_points = np.array(sigma_points)

        # propagete
        propagated_sigma_points = propagete_function(sigma_points)

        transformed_mu = np.zeros(self.N)
        transformed_P = np.zeros([self.N, self.N])

        for i in range(0, 2 * self.N + 1):
            transformed_mu += self.wm[i] * propagated_sigma_points[i, :]

        for i in range(0, 2 * self.N + 1):
            transformed_P += self.wc[i] * np.outer(propagated_sigma_points[i] - transformed_mu,
                                                   propagated_sigma_points[i] - transformed_mu)
        return transformed_mu, transformed_P

    def update(self, observerd_data):
        # predict state
        # compute predicted mean and predicted covariance through dynamic model.
        self.mu, self.P = self.unscented_transform(self.mu, self.P, self.f)

        # update

        # make sigma points

        # poropagete sigma points through measurement model

        # compute the predicted mean, predicted covariance and cross covariance of the state and the measurement

        # compute the filter gain, the filterd state mean and the covariance

        pass
