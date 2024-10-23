import numpy as np


# starting next layer of input layer  belongs to hidden layer

# Dense layer
class Layer_Dense:
 # Layer initialization
 def __init__(self, n_inputs, n_neurons):
 # Initialize weights and biases
   self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # doesnt need transpose of weight
   self.biases = np.zeros((1, n_neurons))

 # Forward pass
 def forward(self, inputs):
   #remember inputs values
   self.inputs = inputs
 # Calculate output values from inputs, weights and biases
   self.output = np.dot(inputs, self.weights) + self.biases

 def backward(self,dvalues):
   # Gradients on parameters
   self.dweights = np.dot(self.inputs.T, dvalues)
   self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
   # Gradient on values
   self.dinputs = np.dot(dvalues, self.weights.T)