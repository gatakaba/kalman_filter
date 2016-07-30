# coding:utf-8
# カルマンフィルタの実装
# このバージョンでは予め遷移行列・観測行列を指定する必要があります
# このプログラムは与えられた時系列に対し、n step後の時系列を確率的に推論することができます
#

import numpy as np


class KalmanFilter(object):
    def __init__(self, transition_matrix, transition_covariance, observe_matrix, transition, observe_covariance,
                 init_state):
        self.state_dim = init_state.shape[0]
        # 遷移行列
        self.A = transition_matrix
        # 観測行列
        self.C = observe_matrix
        # 遷移確率の分散行列
        self.Gamma = observe_covariance
        # 観測確率の分散行列
        self.Sigma = observe_covariance
        # 事前状態ベクトル
        self.priori_state = init_state
        # 事前共分散行列

        self.priori_state = np.ranodm.normal(size=0)
        self.priori_covariance = np.random.normal(size=self.Gamma.shape)
        # カルマンゲイン
        self.kalman_gain = None
        # 事後状態ベクトル
        self.posteriori_state = self.prior_state
        # 事後共分散行列
        self.posteriori_covariance = self.prior_covariance

    def predict(self, observerd_data, n):
        pass
