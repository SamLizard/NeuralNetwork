# Base class
class Layer:
    def __init__(self):
        self.inputs = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, inputs):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
