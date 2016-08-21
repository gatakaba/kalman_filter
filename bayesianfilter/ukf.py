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

        self.state_covariance = np.eye(self.N)
        self.state_mean = np.random.multivariate_normal(np.zeros(self.N), self.state_covariance)

        alpha, beta, kappa = 1, 0, 1

        self.scaling_param = alpha ** 2 * (self.N + kappa) - self.N

        self.wm, self.wc = self.get_weights(alpha, beta)
        self.check_parameter_size()

    @property
    def state_dim(self):
        return self.N

    @property
    def observation_dim(self):
        return self.M

    def check_parameter_size(self):
        pass

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

    def update(self, observed_data):
        # predict state
        # compute predicted mean and predicted covariance through dynamic model.

        predicted_state_mean, predicted_state_covariance = self.unscented_transform(self.state_mean,
                                                                                    self.state_covariance, self.f)
        predicted_state_covariance += self.Q

        observation_mean, observation_covariance = self.unscented_transform(predicted_state_mean,
                                                                            predicted_state_covariance, self.h)
        observation_covariance += self.R

        # compute the cross covariance

        state_sigma_points = self.get_sigma_points(predicted_state_mean, predicted_state_covariance)
        observation_sigma_points = []
        for state_sigma_point in state_sigma_points:
            observation_sigma_points.append(self.h(state_sigma_point))

        cross_covariance = np.zeros([self.N, self.M])

        for i in range(0, 2 * self.N + 1):
            cross_covariance += self.wc[i] * np.outer(state_sigma_points[i] - predicted_state_mean,
                                                      observation_sigma_points[i] - observation_mean)

        K = cross_covariance @ np.linalg.inv(observation_covariance)
        self.state_mean = predicted_state_mean + K @ (observed_data - observation_mean)
        self.state_covariance = predicted_state_covariance - K @ observation_covariance @ K.T
