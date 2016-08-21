# -*- coding: utf-8 -*-
import os, sys, unittest
import numpy as np
from bayesianfilter.ukf import UnscentedKalmanFilter


class UnscentedKalmanFilterTest(unittest.TestCase):
    # テストメソッドを実行するたびに呼ばれる
    def setUp(self):
        def transition_function(x):
            return x

        def observation_function(x):
            return x

        state_dim = 3
        observation_dim = 3

        Q = np.eye(state_dim)
        R = np.eye(observation_dim)
        f = transition_function
        h = observation_function

        self.ukf = UnscentedKalmanFilter(state_dim, observation_dim, f, h, Q, R)

    def test_dim(self):
        self.assertEqual(self.ukf.state_dim, 3)
        self.assertEqual(self.ukf.observation_dim, 3)

    def test_fitering(self):
        x = np.array([2, 2, 2])
        self.ukf.update(x)


if __name__ == '__main__':
    unittest.main()
