import numpy as np

class Activation_ReLU:
 # Forward pass
    def forward(self, inputs):
        # remember  input values
        self.inputs = inputs
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient values where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        # Calculate gradient for input values
        # grad_input = np.array(self.output > 0, dtype=float) * grad_output
        # return grad_input