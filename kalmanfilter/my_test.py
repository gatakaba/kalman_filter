# coding:utf-8
from kalmanfilter import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

dt = 10 ** -1
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
P = np.eye(4) * 10

# set parameter
kf = KalmanFilter(4, 2)

kf.F = A
kf.H = H

kf.Q = Q
kf.R = R

kf.m = m
kf.P = P

t = np.linspace(0, 30, 1000)
trajectory = np.c_[np.cos(t) + t, np.sin(t) * t]

trajectory += np.random.normal(0, 1, size=trajectory.shape)
state_list = []
var_list = []

for y in trajectory:
    m, P = kf.update(y)
    state_list.append(np.copy(m))
    var_list.append(np.copy(P[0, 0]))

X = np.array(state_list)
v = np.array(var_list)

plt.scatter(X[:, 0], X[:, 1], s=v ** 0.5 * 10, alpha=0.25)

plt.plot(trajectory[:, 0], trajectory[:, 1], "bo", label="observerd")
plt.plot(X[:, 0], X[:, 1], "r-", linewidth=5, label="estimated")
plt.legend()

# plt.show()
