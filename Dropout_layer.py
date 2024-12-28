import numpy as np

class Layer_Dropout:
    #initializzze drop out
    def __init__(self, rate):
        # store the dropout rate, invert it to get the success rate
        # for example for a dropout rate of 0.1 we need a success rate of 0.9
        self.rate =1 -  rate

    def forward(self,inputs):
        #save input values
        self.inputs = inputs
        #generate and save the scaled binary mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        #apply mask to output values
        self.output = inputs * self.binary_mask

    # backward
    def backward(self, dvalues):
        #apply the mask to gradients
        self.dinputs = dvalues * self.binary_mask
        

