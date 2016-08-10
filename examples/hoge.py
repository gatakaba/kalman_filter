# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from kalmanfilter.kalmanfilter import KalmanFilter

np.random.seed(0)

dt = 10 ** -3
M, D, K = 1, 0, 0

A = np.array([[1, dt],
              [-K / M * dt, -D / M * dt + 1]])

B = np.array([[0, 0], [0, dt / M]])

C = np.array([1, 0])
C = np.atleast_2d(C)

Q = np.eye(2) * 0.01
R = np.eye(1)

s = np.array([0, 0])

kf = KalmanFilter(A, C, Q, R, s, initial_covariance=None, drive_matrix=B)
j = []
l = []
k = []
N = 1000
x = 0
t = np.arange(0, N * dt, dt)

for i in range(N):
    u = np.array([0, -M * 9.8])

    x = -t[i] ** 2 * 9.8 / 2

    kf.update(x + np.random.normal(0, 1), u)
    j.append(np.copy(x))
    l.append(np.copy(kf.current_state[0])[0])
    k.append(np.copy(kf.current_state[1])[0, 0])

true_x = np.array(j)
x = np.array(l)
v = np.array(k)

plt.plot(t, true_x, "g-")
plt.fill_between(t, x - v ** 0.5, x + v ** 0.5, alpha=0.25)
plt.plot(t, x, "ro-")

plt.show()
