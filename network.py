import numpy as np

from layer import Layer


def calculate_error(outputs: np.ndarray, answers: np.ndarray) -> np.ndarray:
    return (answers - outputs) ** 2


class Network:
    def __init__(self, inputs: int, outputs: int, hidden_layers: int, hidden_neurons: int):
        self.layers = [Layer(inputs, hidden_neurons)] + \
                      [Layer(hidden_neurons, hidden_neurons) for _ in range(hidden_layers - 1)] + \
                      [Layer(hidden_neurons, outputs)]

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        last_output = inputs
        for layer in self.layers:
            last_output = layer.feed(last_output)

        return last_output

    def back_propagate(self, errors: np.ndarray):
        for layer in reversed(self.layers):
            layer.actualize(errors)
            errors = layer.errors(errors)

    def train(self, inputs: np.ndarray, answers: np.ndarray):
        outputs = self.feed_forward(inputs)
        errors = calculate_error(outputs, answers)
        self.back_propagate(errors)


if __name__ == '__main__':
    # define the network
    network = Network(inputs=2, outputs=1, hidden_layers=2, hidden_neurons=2)

    # define the training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    # test the network
    test_cases = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    for test_case in test_cases:
        output = network.feed_forward(np.array(test_case))
        print(f"Input: {test_case} Output: {output}")

    # train the network
    for i in range(10000):
        network.train(X, y)

    for test_case in test_cases:
        output = network.feed_forward(np.array(test_case))
        print(f"Input: {test_case} Output: {output}")
