import numpy as np

from network import NeuralNetwork


def main():
    input_data = np.fromfile("data/train-images-idx3-ubyte", np.uint8, offset=16)
    input_data = input_data.reshape(60000, 784)
    input_data = input_data.astype('float32')
    input_data /= 255

    output_data = np.fromfile("data/train-labels-idx1-ubyte", np.uint8, offset=8)
    output_data = np.array([np.ravel(np.eye(1, 10, k=n).T) for n in output_data])
    output_data = output_data.astype(int)

    nn = NeuralNetwork(inputs=28 ** 2, outputs=10, hidden_neurons=16, hidden_layers=2)
    nn.train(input_data, output_data, iterations=5)

    test_data = np.fromfile("data/t10k-images-idx3-ubyte", np.uint8, offset=16)
    test_data = test_data.reshape(10000, 784)
    test_data = test_data.astype('float32')
    test_data /= 255

    test_output = np.fromfile("data/t10k-labels-idx1-ubyte", np.uint8, offset=8)
    # test_output = np.array([np.ravel(np.eye(1, 10, k=n).T) for n in output_data])
    test_output = test_output.astype(int)

    out = nn.predict(test_data[0])
    print(out)
    print(test_output[0])


def predict(nn: NeuralNetwork, test_input: np.ndarray, threshold: float = 0.5):
    output = nn.predict(test_input)

    certainty = output.ravel()[0]
    if output >= threshold:
        print(f"Output is 1 with {certainty * 100:.2f}% certainty")
    else:
        print(f"Output is 0 with {100 - certainty * 100:.2f}% certainty")


if __name__ == '__main__':
    main()
