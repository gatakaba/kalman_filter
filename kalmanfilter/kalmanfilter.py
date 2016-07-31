# coding:utf-8
# カルマンフィルタの実装

import numpy as np


class KalmanFilter(object):
    def __init__(self, dim_x, dim_z):
        """dim is the number of dimension"""

        self.z  # 状態
        self.F  # 遷移行列
        self.H  # 観測行列

        self.Q  # プロセスノイズの分散共分散行列
        self.R  # 観測ノイズの分散共分散行列

        self.K  # カルマンゲイン
        self.m  # 状態推定値
        self.P  # 推定誤差分散共分散行列

    def _update(self, observerd_data):
        """
        観測されたデータに応じて、状態と推定共分散行列を更新する
        :param observerd_data (1d ndarray): 観測値
        :return: self
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
        return self
