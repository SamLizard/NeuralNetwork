import numpy as np

LEARNING_RATE = 0.1


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, input_shape: int, neurons: int):
        self.weights = np.random.rand(neurons, input_shape)
        self.bias = np.zeros((neurons, 1))
        self.input = None
        self.output = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = sigmoid(self.weights @ input_data + self.bias)
        return self.output

    def update(self, error: np.ndarray):
        self.weights -= LEARNING_RATE * (error @ self.input.T)
        self.bias -= LEARNING_RATE * np.reshape(np.mean(error, axis=1), self.bias.shape)
