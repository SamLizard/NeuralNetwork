import numpy as np

from network import NeuralNetwork


def main():
    input_data = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
    output_data = np.array([[0],
                            [1],
                            [1],
                            [0]])

    nn = NeuralNetwork(inputs=2, outputs=1, hidden_neurons=4, hidden_layers=2)
    nn.train(input_data, output_data)

    test = np.array([[1, 0]])
    predict(nn, test)
    test = np.array([[0, 0]])
    predict(nn, test)
    test = np.array([[0, 1]])
    predict(nn, test)
    test = np.array([[1, 1]])
    predict(nn, test)


def predict(nn: NeuralNetwork, test_input: np.ndarray, threshold: float = 0.5):
    output = nn.predict(test_input)

    certainty = output.ravel()[0]
    if output >= threshold:
        print(f"Output is 1 with {certainty * 100:.2f}% certainty")
    else:
        print(f"Output is 0 with {100 - certainty * 100:.2f}% certainty")


if __name__ == '__main__':
    main()
