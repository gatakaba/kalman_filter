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
        self.P  # 推定誤差分散共分散行列

    def update(self, observerd_data, n):
        pass
