# coding :utf-8
import numpy as np
import matplotlib.pyplot as plt

dt = 10 ** -2
q1, q2 = 1, 1  # process noise
sigma1, sigma2 = 0.5, 0.5  # measurement noise

trajectory = np.load("log.npy")

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
P = np.eye(4)

state_list = []
var_list = []
for y in trajectory:
    # prediction step
    m = A @ m
    P = A @ P @ A.T + Q

    # update step
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    m = m + K @ (y - H @ m)
    P = P - K @ S @ K.T
    state_list.append(np.copy(m))
    var_list.append(np.array(P[0, 0]))

X = np.array(state_list)
v = np.array(var_list)
print(X[:4, :])

plt.scatter(X[:, 0], X[:, 1], s=v * 10000, alpha=0.25)
plt.plot(X[:, 0], X[:, 1], "r-")
plt.scatter(trajectory[:, 0], trajectory[:, 1], s=1)
plt.show()
