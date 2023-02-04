import numpy as np

from layer import Layer


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)


def calculate_error(previous_weights: np.ndarray, previous_error: np.ndarray, layer_output: np.ndarray) -> np.ndarray:
    return (previous_weights.T @ previous_error) * sigmoid_derivative(layer_output)


class NeuralNetwork:
    def __init__(self, inputs: int, outputs: int, hidden_neurons: int, hidden_layers: int):
        self.layers = [Layer(inputs, hidden_neurons)] + \
                      [Layer(hidden_neurons, hidden_neurons) for _ in range(hidden_layers - 1)] + \
                      [Layer(hidden_neurons, outputs)]

    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]

    def forward_prop(self, input_data: np.ndarray):
        for layer in self.layers:
            input_data = layer.forward(input_data)

    def back_prop(self, output_data: np.ndarray):
        error = 2 * (self.output_layer.output - output_data) / output_data.size
        self.output_layer.update(error)

        for i, layer in reversed(list(enumerate(self.layers[:-1]))):
            error = calculate_error(self.layers[i + 1].weights, error, layer.output)
            layer.update(error)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        input_data = input_data.reshape((input_data[0].size, 1))
        self.forward_prop(input_data)
        return self.output_layer.output

    def train(self, input_data: np.ndarray, output_data: np.ndarray, iterations: int = 10000):
        # input_data = input_data.T
        # output_data = output_data.T
        input_shape = input_data[0].size
        output_shape = output_data[0].size

        for _ in range(iterations):
            for i, o in zip(input_data, output_data):
                # indices = np.random.randint(0, 60000, batch_size)
                # batch_input = input_data.T[indices, :].T
                # batch_output = output_data.T[indices, :].T
                i = i.reshape((input_shape, 1))
                o = o.reshape((output_shape, 1))
                self.forward_prop(i)
                self.back_prop(o)
