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

Q = np.eye(2) * 0.001
R = np.eye(1)

s = np.array([0, 10])

kf = KalmanFilter(A, C, Q, R, s, initial_covariance=None, drive_matrix=B)
j = []
l = []
k = []
N = 2000
x = 0
t = np.arange(0, N * dt, dt)
cnt = 0

y = -t ** 2 * 9.8 / 2 + 10 * t
for i in range(N):
    x = y[i]
    u = np.array([0, -M * 9.8])

    if i < 500:
        kf.update(x + np.random.normal(0, 1), u)
        j.append(np.copy(x))
        l.append(np.copy(kf.current_state[0])[0])
        k.append(np.copy(kf.current_state[1])[0, 0])
    else:
        m, p = kf.predict_state(cnt + 1, u, spot_estimation=True)
        j.append(np.copy(x))
        l.append(np.copy(m)[0])
        k.append(np.copy(p)[0, 0])
        cnt += 1

true_x = np.array(j)
x = np.array(l)
v = np.array(k)

plt.plot(t, true_x, "g-")
plt.fill_between(t, x - v ** 0.5, x + v ** 0.5, alpha=0.25)
plt.plot(t, x, "ro-")

plt.show()
