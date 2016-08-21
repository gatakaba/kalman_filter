# coding:utf-8
import numpy as np
from bayesianfilter import UnscentedKalmanFilter

np.random.seed(11
               )

dt = 0.01

A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

sensor_positions = [[-3, 5],
                    [0, 10],
                    [3, 10],
                    [-1, 3],
                    [3, -5]]

Q = np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0],
              [0, dt ** 3 / 3, 0, dt ** 2 / 2],
              [dt ** 2 / 2, 0, dt, 0],
              [0, dt ** 2 / 2, 0, dt]])

R = np.eye(len(sensor_positions))

Q *= 0.1
R *= 0.01


def transition_function(x):
    return A @ x


def observation_function(x):
    observation_list = []
    for sensor_position in sensor_positions:
        theta = np.arctan2(x[1] - sensor_position[1], x[0] - sensor_position[0])
        observation_list.append(theta)

    return np.array(observation_list)


true_state_list = []
observation_list = []

state = np.array([0, 0, 0, 0])

for i in range(1000):
    state = np.random.multivariate_normal(transition_function(state), Q)
    observation = np.random.multivariate_normal(observation_function(state), R)

    true_state_list.append(state)
    observation_list.append(observation)

ukf = UnscentedKalmanFilter(state_dim=4, observation_dim=len(sensor_positions), transition_function=transition_function,
                            observation_function=observation_function, process_noise=Q, observation_noise=R)

estimate_list = []
for i, observation in enumerate(observation_list):
    ukf.update(observation)
    estimate_list.append(ukf.state_mean)

estimate_list = np.array(estimate_list)

import matplotlib.pyplot as plt

plt.suptitle("tracking using Wiener velocity model")
plt.subplot(211)
plt.title("obserevd data")
plt.plot(observation_list)
plt.ylabel("angle [rad]")
plt.subplot(212)
plt.title("trajectory")
plt.plot(np.array(true_state_list)[:, 0], np.array(true_state_list)[:, 1], "bo-", alpha=0.5, label="true")
plt.plot(estimate_list[:, 0], estimate_list[:, 1], "ro-", alpha=0.5, label="estimate")

plt.scatter(np.array(sensor_positions)[:, 0], np.array(sensor_positions)[:, 1], marker="*", label="sensor")
plt.legend()

plt.show()
