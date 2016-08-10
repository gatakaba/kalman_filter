# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from kalmanfilter.kalmanfilter import KalmanFilter

np.random.seed(0)

dt = 10 ** -3
M, D, K = 10, 0, 0

A = np.array([[1, dt],
              [-K / M * dt, -D / M * dt + 1]])

B = np.array([[0, 0], [0, dt / M]])
C = np.atleast_2d([1, 0])
q = 1
r = 1
Q = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]]) * q
R = np.eye(1) * r
s = np.array([0, 0])
kf = KalmanFilter(A, C, Q, R, s, initial_covariance=None, drive_matrix=B)

N = 2000
t = np.arange(0, N * dt, dt)
true_x = np.empty(N)
observed_x = np.empty(500)
estimated_x = np.empty(N)
estimated_variance = np.empty(N)

y = -t ** 2 * 9.8 / 2 + 10 * t + 10

for i in range(500):
    x = y[i] + np.random.normal(0, 0.5)
    observed_x[i] = x
    u = np.array([0, -M * 9.8])
    kf.update(x, u)
    estimated_x[i] = kf.current_state[0][0]
    estimated_variance[i] = kf.current_state[1][0, 0]

m, p = kf.predict_state(N - 500, u)

for i in range(N - 500):
    estimated_x[i + 500] = m[i][0]
    estimated_variance[i + 500] = p[i][0, 0]

plt.plot(t[:500], observed_x, "k-", label="observed trajectory", alpha=0.25)
plt.plot(t, y, "g-", label="true trajectory")
plt.fill_between(t, estimated_x - estimated_variance ** 0.5, estimated_x + estimated_variance ** 0.5, alpha=0.25)
plt.plot(t, estimated_x, "ro-", label="filterd trajectory")
plt.legend()
plt.show()
