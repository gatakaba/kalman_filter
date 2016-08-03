# coding:utf-8
"""
# カルマンフィルタの実装

## モデル
    状態方程式
        * p(z_{n+1}|z_{n}) = N(z_{n+1}|F z_{n},\gamma)
    観測方程式
        * p(x_{n}|z_{n}) = N(x_{n}|C z_{n},\sigma)
## 機能
    * 状態推定(フィルタリング)
    * 状態予測
    * 観測値の予測
"""
import numpy as np


class KalmanFilter(object):
    def __init__(self, transition_matrix, observation_matrix, process_noise, observation_noise):
        self.F = transition_matrix  # 遷移行列
        self.H = observation_matrix  # 観測行列

        self.Q = process_noise  # プロセスノイズの分散共分散行列
        self.R = observation_noise  # 観測ノイズの分散共分散行列

        self.m = np.random.normal(size=4)  # 状態推定値
        self.P = np.eye(4)  # 推定誤差分散共分散行列

        self.K = None  # カルマンゲイン

    @property
    def get_state(self):
        return self.m, self.P

    @property
    def transition_matrix(self):
        return self.F

    @property
    def observation_matrix(self):
        return self.H

    @property
    def process_covariance_matrix(self):
        return self.Q

    @property
    def observation_covariance_matrix(self):
        return self.R

    def update(self, observerd_data):
        """
        p(z_{t}|x_{1:t})
        観測されたデータに応じて、状態と推定共分散行列を更新する
        :param 観測値 (1d ndarray)
        :return: 状態の確率密度関数の平均値と分散
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

    def predict_state(self, k):
        # p(z_{t+k}|x_{1:k})
        # estimate state after k step
        pass

    def predict_observation(self, k):
        # p(x_{t+k}|x_{1:k})
        # estimate observation after k step
        pass
