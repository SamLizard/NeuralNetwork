import numpy as np
from colorama import just_fix_windows_console, Fore  # noqa

from activations import Activation
from layers import FCLayer, ActivationLayer
from network import Network


def main():
    x_test, x_train, y_test, y_train = prepare_dataset()

    # Network
    net = Network([FCLayer(28 * 28, 100),
                   ActivationLayer(Activation.TANH),
                   FCLayer(100, 50),
                   ActivationLayer(Activation.TANH),
                   FCLayer(50, 10),
                   ActivationLayer(Activation.TANH)])

    # train on 1000 samples
    # as we didn't implement mini-batch GD,
    # training will be pretty slow if we update at each iteration on 60000 samples...
    net.fit(x_train[:1000], y_train[:1000], epochs=35, learning_rate=0.1)

    display_result_colored(net.predict(x_test[0:20]), y_test[:20])


def prepare_dataset():
    x_train = np.fromfile("data/train-images-idx3-ubyte", np.uint8, offset=16)
    x_train = x_train.reshape(60000, 1, 28 * 28)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = np.fromfile("data/train-labels-idx1-ubyte", np.uint8, offset=8)
    y_train = np.array([np.ravel(np.eye(1, 10, k=n).T) for n in y_train])
    y_train = y_train.astype(int)

    x_test = np.fromfile("data/t10k-images-idx3-ubyte", np.uint8, offset=16)
    x_test = x_test.reshape(10000, 1, 28 * 28)
    x_test = x_test.astype('float32')
    x_test /= 255

    y_test = np.fromfile("data/t10k-labels-idx1-ubyte", np.uint8, offset=8)
    y_test = y_test.astype(int)
    return x_test, x_train, y_test, y_train


def display_result_colored(arrays: np.ndarray, expected_outputs: np.ndarray):
    for array, output in zip(arrays, expected_outputs):
        certainty = np.max(array)
        guessed_number = np.argmax(array)
        print(f"{Fore.GREEN if guessed_number == output else Fore.RED}{guessed_number}{Fore.RESET}", end=" ")
        print("with", end=" ")
        print(f"{Fore.GREEN if certainty >= 0.7 else Fore.RED}{certainty * 100:.2f}%{Fore.RESET}", end=" ")
        print("certainty. Expected number:", end=" ")
        print(f"{Fore.BLUE}{output}")


if __name__ == '__main__':
    just_fix_windows_console()
    main()
