from activations import Activation
from .layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation: Activation = Activation.RELU):
        super().__init__()
        self.activation, self.activation_prime = activation.value

    # returns the activated input
    def forward_propagation(self, inputs):
        self.inputs = inputs
        self.output = self.activation(self.inputs)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.inputs) * output_error
