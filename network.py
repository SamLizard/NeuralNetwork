from typing import List

import numpy as np

from layers import Layer
from loss import mse, mse_prime


class Network:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.loss = mse
        self.loss_prime = mse_prime

    # set loss to use
    def use_loss(self, f, f_prime):
        self.loss = f
        self.loss_prime = f_prime

    def forward_propagation(self, inputs: np.ndarray):
        for layer in self.layers:
            inputs = layer.forward_propagation(inputs)

        return inputs

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        return [self.forward_propagation(inputs) for inputs in input_data]

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for inputs, outputs in zip(x_train, y_train):
                output = self.forward_propagation(inputs)

                # compute loss (for display purpose only)
                err += self.loss(outputs, output)

                # backward propagation
                error = self.loss_prime(outputs, output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print(f'epoch [{i + 1:>5}/{epochs}] error={err:f}')

            if err < 0.0002:
                break
