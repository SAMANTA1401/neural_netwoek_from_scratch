import numpy as np
from Dense_Layer import Layer_Dense
from ReLU_Act import Activation_ReLU
from Softmax_Act import Activation_Softmax
from Cat_Crossentropy_Loss import Loss_Categorical_Crossentropy


class Neural_Network:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.layers = []

    def add_layer(self,layer):
        self.layers.append(layer)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values
    dense2 = Layer_Dense(3, 3)
    # Create Softmax activation (to be used with Dense layer):
    activation2 = Activation_Softmax()
    # Create loss function
    loss_function = Loss_Categorical_Crossentropy()
    # Perform a forward pass of our training data through this layer




    dense1.forward(self.X)
    # Perform a forward pass through activation function
    # it takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # it takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through activation function
    # it takes the output of second dense layer here
    activation2.forward(dense2.output)
    # Let's see output of the first few samples:
    print(activation2.output[:5])
    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)
    # Print loss value
    print('loss:', loss)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    # predictions = np.argmax(activation2.output, axis=1)
    # if len(y.shape) == 2:
    #  y = np.argmax(y, axis=1)
    # accuracy = np.mean(predictions == y)
    # # Print accuracy
    # print('acc:', accuracy)
