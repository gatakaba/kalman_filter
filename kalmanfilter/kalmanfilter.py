# coding:utf-8
# カルマンフィルタの実装

import numpy as np


class KalmanFilter(object):
    def __init__(self, dim_x, dim_z):
        self.z = None  # 状態
        self.F = None  # 遷移行列
        self.H = None  # 観測行列

        self.Q = None  # プロセスノイズの分散共分散行列
        self.R = None  # 観測ノイズの分散共分散行列

        self.K = None  # カルマンゲイン
        self.m = None  # 状態推定値
        self.P = None  # 推定誤差分散共分散行列

    def _update(self, observerd_data):
        """
        観測されたデータに応じて、状態と推定共分散行列を更新する
        :param observerd_data (1d ndarray): 観測値
        :return: 状態の確率密度関数の平均値
        """

        x = observerd_data
        F, H = self.F, self.H
        Q, R = self.Q, self.R
        m = np.copy(self.m)
        P = np.copy(self.P)
        # prediction step
        m = F @ m
        P = F @ P @ F.T + Q
        # update step
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ (x - H @ m)
        P = P - K @ S @ K.T
        # update parameter
        self.m = m
        self.P = P
        return self.m


if __name__ == "__main__":
    dt = 10 ** -3
    sigma1 = 0.5
    sigma2 = 0.5
    q1 = 0.5
    q2 = 0.5
    # transition matrix
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # process noise covariance
    Q = np.array([[q1 * dt ** 3 / 3, 0, q1 * dt ** 2 / 2, 0],
                  [0, q1 * dt ** 3 / 3, 0, q2 * dt ** 2 / 2],
                  [q1 * dt ** 2 / 2, 0, q1 * dt, 0],
                  [0, q2 * dt ** 2 / 2, 0, q2 * dt]])

    # observe matrix
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    # measurement noise covariance
    R = np.array([[sigma1 ** 2, 0],
                  [0, sigma2 ** 2]])

    m = np.random.normal(size=4)
    P = np.eye(4)

    # set parameter
    kf = KalmanFilter(4, 2)

    kf.F = A
    kf.H = H

    kf.Q = Q
    kf.R = R

    kf.m = m
    kf.P = P

    trajectory = np.random.normal(size=[1000, 2])
    state_list = []
    var_list = []

    for y in trajectory:
        state_list.append(np.copy(m))
        var_list.append(np.array(P[0, 0]))

    X = np.array(state_list)
    v = np.array(var_list)
    print(X)
