# coding:utf-8
# カルマンフィルタの実装
# このバージョンでは予め遷移行列・観測行列を指定する必要があります
# このプログラムは与えられた時系列に対し、n step後の時系列を確率的に推論することができます
#

import numpy as np


class KalmanFilter(object):
    def __init__(self, transition_matrix, transition_covariance, observe_matrix, transition, observe_cobariance,
                 init_state):
        # 遷移行列
        self.A = transition_matrix
        # 観測行列
        self.C = observe_matrix
        # 遷移確率の分散行列
        self.Gamma = observe_cobariance
        # 観測確率の分散行列
        self.Sigma = observe_cobariance
        # 初期状態
        self.state = init_state

    def fit(self):
        pass

    def predict(self, n):
        pass
