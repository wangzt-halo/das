import numpy as np


def world2pixel(X, K, R, t):

    x = np.dot(R, X) + t

    x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x


def pixel2world(x, K, R, t):
    X = x.copy()
    X[0, :] = X[0, :] - K[0, 2]
    X[1, :] = X[1, :] - K[1, 2]
    X[:2] = np.dot(np.linalg.inv(K[:2, :2]), X[:2])
    x1 = X.copy()
    X[0:2, :] = X[0:2, :] * X[2, :]
    x2 = X.copy()
    X = np.dot(np.linalg.inv(R), (X - t))
    x3 = X.copy()
    return x1, x2, x3


