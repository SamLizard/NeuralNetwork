from enum import Enum

import numpy as np


# activation function and its derivative
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return (x > 0).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Activation(Enum):
    TANH = (tanh, tanh_prime)
    RELU = (relu, relu_prime)
    SIGMOID = (sigmoid, sigmoid_prime)
