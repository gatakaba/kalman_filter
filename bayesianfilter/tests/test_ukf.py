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
        observation_dim = 2

        Q = np.eye(state_dim)
        R = np.eye(observation_dim)
        f = transition_function
        h = observation_function

        self.ukf = UnscentedKalmanFilter(state_dim, observation_dim)

    def test_dim(self):
        self.assertEqual(self.kf.state_dim, 3)
        self.assertEqual(self.kf.observation_dim, 2)

    def test_fitering(self):
        x = np.array([2, 2])
        self.kf.update(x)

    def test_predict_observation(self):
        x = np.array([2, 2])
        self.kf.update(x)
        self.assertEqual(len(self.kf.predict_observation(3)[0]), 3)


if __name__ == '__main__':
    unittest.main()
