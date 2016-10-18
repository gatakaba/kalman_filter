# -*- coding: utf-8 -*-
import os, sys, unittest
import numpy as np
from kalmanfilter.kalmanfilter import KalmanFilter


class KalmanFilterTest(unittest.TestCase):
    # テストメソッドを実行するたびに呼ばれる
    def setUp(self):
        A = np.eye(3)
        C = np.random.normal(size=[2, 3])
        B = np.random.normal(size=[3, 2])
        Q = np.eye(3)
        R = np.eye(2)
        self.kf = KalmanFilter(A, C, Q, R, drive_matrix=B)

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
