import numpy as np
from colorama import Fore

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
    nn.train(input_data, output_data, iterations=1)

    test_data = np.fromfile("data/t10k-images-idx3-ubyte", np.uint8, offset=16)
    test_data = test_data.reshape(10000, 784)
    test_data = test_data.astype('float32')
    test_data /= 255

    test_output = np.fromfile("data/t10k-labels-idx1-ubyte", np.uint8, offset=8)
    # test_output = np.array([np.ravel(np.eye(1, 10, k=n).T) for n in output_data])
    test_output = test_output.astype(int)

    predict(nn, test_data[:50], test_output[:50])

    print(f"The neural network accuracy is: {nn.accuracy(test_data, test_output) * 100:.2f}%")


def predict(nn: NeuralNetwork, test_inputs: np.ndarray, expected_outputs: np.ndarray):
    output = np.array([nn.predict(i) for i in test_inputs])
    display_result_colored(output, expected_outputs)


def display_result_colored(arrays: np.ndarray, expected_outputs: np.ndarray):
    for array, output in zip(arrays, expected_outputs):
        certainty = np.max(array)
        guessed_number = np.argmax(array)
        print(f"{Fore.GREEN if guessed_number == output else Fore.RED}{guessed_number}{Fore.RESET}", end=" ")
        print("with", end=" ")
        print(f"{Fore.GREEN if certainty >= 0.7 else Fore.RED}{certainty * 100:.2f}%{Fore.RESET}", end=" ")
        print("certainty. Expected number:", end=" ")
        print(f"{Fore.BLUE}{output}{Fore.RESET}")


if __name__ == '__main__':
    main()
