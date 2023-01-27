from typing import List

import numpy as np

LEARNING_RATE = 0.1


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


class Layer:
    def __init__(self, input_shape: int, neurons: int):
        self.weights = np.random.rand(neurons, input_shape)
        self.bias = np.zeros((neurons, 1))
        self.output = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.output = sigmoid(self.weights @ input_data + self.bias)
        return self.output

    def update(self, layer_input: np.ndarray, input_shape: int, error: np.ndarray):
        self.weights -= LEARNING_RATE * ((error @ layer_input.T) / input_shape)
        self.bias -= LEARNING_RATE * np.reshape(np.mean(error, axis=1), self.bias.shape)


class NeuralNetwork:
    def __init__(self, input_shape: int, output_shape: int, hidden_neurons: int):
        self.input_shape = input_shape
        self.layers = [Layer(input_shape, hidden_neurons),
                       Layer(hidden_neurons, output_shape)]

    @property
    def input_layer(self) -> Layer:
        return self.layers[0]

    @property
    def hidden_layers(self) -> List[Layer]:
        return self.layers[1:-1]

    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]

    def forward_prop(self, input_data: np.ndarray):
        for layer in self.layers:
            input_data = layer.forward(input_data)

    def back_prop(self, input_data: np.ndarray, output_data: np.ndarray):
        inputs = [input_data] + [layer.output for layer in self.hidden_layers]
        outputs = [self.input_layer.output] + [layer.output for layer in self.hidden_layers]

        error = self.output_layer.output - output_data
        self.output_layer.update(outputs[-1], self.input_shape, error)

        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            error = (self.layers[i + 1].weights.T @ error) * layer.output * (1 - layer.output)
            layer.update(inputs[i - 1], self.input_shape, error)

        error = (self.layers[1].weights.T @ error) * self.input_layer.output * (1 - self.input_layer.output)
        self.input_layer.update(input_data, self.input_shape, error)

    def predict(self, input_data: np.ndarray, threshold: float = 0.5):
        self.forward_prop(input_data)

        certainty = self.output_layer.output.ravel()[0]
        if self.output_layer.output >= threshold:
            print(f"Output is 1 with {certainty * 100:.2f}% certainty")
        else:
            print(f"Output is 0 with {100 - certainty * 100:.2f}% certainty")

    def train(self, input_data: np.ndarray, output_data: np.ndarray, iterations: int = 5000):
        for i in range(iterations):
            self.forward_prop(input_data)
            self.back_prop(input_data, output_data)


def main():
    input_data = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    output_data = np.array([[0, 1, 1, 0]])

    nn = NeuralNetwork(input_data.shape[0], output_data.shape[0], 2)
    nn.train(input_data, output_data)

    test = np.array([[1], [0]])
    nn.predict(test)
    test = np.array([[0], [0]])
    nn.predict(test)
    test = np.array([[0], [1]])
    nn.predict(test)
    test = np.array([[1], [1]])
    nn.predict(test)


if __name__ == '__main__':
    main()
