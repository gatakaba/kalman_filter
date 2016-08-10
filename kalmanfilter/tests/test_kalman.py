# -*- coding: utf-8 -*-
import os, sys, unittest
import numpy as np
from kalmanfilter.kalmanfilter import KalmanFilter


class KalmanFilterTest(unittest.TestCase):
    # テストメソッドを実行するたびに呼ばれる
    def setUp(self):
        F = np.random.normal(size=[3, 3])
        H = np.random.normal(size=[2, 3])

        Q = np.eye(3)
        R = np.eye(2)
        self.kf = KalmanFilter(F, H, Q, R)

    def test_dim(self):
        self.assertEqual(self.kf.state_dim, 3)
        self.assertEqual(self.kf.observation_dim, 2)


if __name__ == '__main__':
    unittest.main()
