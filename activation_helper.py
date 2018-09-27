import numpy as np


def sigmoid(x):
    h = 0.0001
    return 1/1+np.exp(-x + h)


def tanh(x):
    h = 0.0001
    return (np.exp(x + h) - np.exp(-x + h)) / ( np.exp(x + h) + np.exp(-x + h))


def relu(x):
    return np.maximum(0, x)


def lrelu(x):
    return np.maximum(0.01 * x, x)