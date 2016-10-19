import numpy as np
from kalmanfilter import KalmanFilter
from kalmansmoother import KalmanSmoother

A = np.array([[1, 0.01],
              [0, 1]])

C = np.atleast_2d([1, 0])

Q = np.array([[0.01, 0], [0, 0.01]])

R = np.eye(1)

ks = KalmanSmoother(A, C, Q, R)
kf = KalmanFilter(A, C, Q, R)

t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, size=len(t))

state_list = []
covariance_list = []

for i in range(1000):
    kf.update(x[i])

    state_list.append(np.array(kf.current_state[0]))
    covariance_list.append(np.array(kf.current_state[1]))

print(np.array(covariance_list).shape)

mu, p = ks.smoothing(x)

import matplotlib.pyplot as plt

plt.plot(t, x, label="observed")
plt.plot(t, mu[:, 0], label="smoothed")
plt.fill_between(t, mu[:, 0] - p[:, 0, 0] ** 0.5, mu[:, 0] + p[:, 0, 0] ** 0.5, alpha=0.25, color="green"
                                                                                                  "",
                 label="smoothed 1σ")

plt.plot(t, np.array(state_list)[:, 0], label="filterd")

plt.fill_between(t, np.array(state_list)[:, 0] - np.array(covariance_list)[:, 0, 0] ** 0.5,
                 np.array(state_list)[:, 0] + np.array(covariance_list)[:, 0, 0] ** 0.5, alpha=0.25, color="red",
                 label="filtered 1σ")

plt.legend()
plt.show()
