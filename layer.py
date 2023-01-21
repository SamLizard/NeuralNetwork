import numpy as np

LEARNING_RATE = 0.01


def sigmoid(x: float):
    return 1 / (1 + np.exp(-x))


sigmoid_v = np.vectorize(sigmoid)


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)


class Layer:
    def __init__(self, last_neurons: int, neurons: int):
        self.weights = np.random.uniform(-1, 1, (last_neurons, neurons))
        self.biases = np.random.uniform(-1, 1, neurons)
        self.feed_forward_inputs = None
        self.feed_forward_outputs = None

    def feed(self, inputs: np.ndarray) -> np.ndarray:
        self.feed_forward_inputs = inputs
        self.feed_forward_outputs = sigmoid_v(np.dot(inputs, self.weights) + self.biases)
        return self.feed_forward_outputs

    def actualize(self, errors: np.ndarray):
        # calculate the gradient of the weights and biases using the errors and sigmoid derivative
        # gradient = LEARNING_RATE * (errors, sigmoid_derivative(self.feed_forward_outputs))
        gradient = LEARNING_RATE * errors * sigmoid_derivative(self.feed_forward_outputs)
        # gradient = LEARNING_RATE * errors * sigmoid_derivative(np.dot(self.feed_forward_inputs, self.weights) + self.biases)

        # update the weights and biases using a learning rate
        self.weights -= np.dot(self.feed_forward_inputs.T, gradient)
        # self.biases += np.sum(gradient, axis=0)

    def errors(self, errors: np.ndarray) -> np.ndarray:
        return np.dot(errors, np.transpose(self.weights))  # maybe use dot instead of *
