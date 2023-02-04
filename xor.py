import pprint

import numpy as np

from activations import Activation
from layers import ActivationLayer, FCLayer
from network import Network


def main():
    # training data
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # network
    net = Network([FCLayer(2, 4),
                   ActivationLayer(Activation.TANH),
                   FCLayer(4, 2),
                   ActivationLayer(Activation.TANH),
                   FCLayer(2, 1),
                   ActivationLayer(Activation.TANH)])

    # train
    net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)

    # test
    out = net.predict(x_train)
    pprint.pprint(out)


if __name__ == '__main__':
    main()
