# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from kalmanfilter.kalmanfilter import KalmanFilter

dt = 10 ** -2
K = 1
C = 1
M = 1

u = -M * 9.8
F = np.array([[1, dt, 0],
              [-K / M * dt, -C / M * dt + 1, dt / M * u],
              [0, 0, 1]])

H = np.array([[1, 0, 0], [0, 0, 0]])

Q = np.eye(3) * 0.01
R = np.eye(2) * 0.01
s = np.array([0, 0, 1])

kf = KalmanFilter(F, H, Q, R, s, initial_covariance=None)
t = np.linspace(0, -100)
for i in range(1000):
    kf.update(np.array([t[i], 0]))
    print(kf.current_state[0])
