from Softmax_Act import Activation_Softmax
from Cat_Crossentropy_Loss import Loss_Categorical_Crossentropy
import numpy as  np

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Crossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)  #dvalues: The derivative of the loss with respect to the output.
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        # dinputs = [[-10,  0,  0],
        #    [ 0, -2,  0],
        #    [ 0,  0, -1.11],
        #    [-10,  0,  0]]
        #     y_true = [0, 1, 2, 0]
        #     samples = 4

        # dinputs = [[-11,  0,  0],
        #    [ 0, -3,  0],
        #    [ 0,  0, -2.11],
        #    [-11,  0,  0]]

        # Normalize gradient
        self.dinputs = self.dinputs / samples

    def regularization_loss(self, layer):
        #0 by default
        regularization_loss = 0
        #L1 regularization - weights
        # Calculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        #L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        #L1 regularization - biases
        # Calculate only when factor greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        #L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss
    
    # calculate the data and regularization losses
    # given model output and ground truth values
    def calculate_loss(self, model_output, y):
        # calculate loss from output of activation2 (softmax activation)
        data_loss = self.forward(model_output, y)
        # calculate regularization penalty
        # regularization_loss = self.regularization_loss(model_output)
        # # calculate total loss
        # total_loss = data_loss + regularization_loss
        # return total_loss
        data_loss = np.mean(data_loss)
        return data_loss
    
