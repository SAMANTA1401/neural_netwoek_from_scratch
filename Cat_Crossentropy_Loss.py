import numpy as np
# Cross-entropy loss
class Loss_Categorical_Crossentropy:
 # Forward pass
  def forward(self, y_pred, y_true):                 
    # Number of samples in a batch
    samples = len(y_pred)
    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Probabilities for target values -
    # only if categorical labels
    if len(y_true.shape) == 1:
      correct_confidences = y_pred_clipped[
      range(samples),
      y_true
      ]
    # Mask values - only for one-hot encoded labels
    elif len(y_true.shape) == 2:
      correct_confidences = np.sum(
      y_pred_clipped*y_true,
      axis=1
      )
    negative_log_likelihoods = -np.log(correct_confidences)
    return negative_log_likelihoods
  # Losses
  def calculate(self,y_pred, y_true):
    correct_confidences = self.forward(y_pred, y_true)
    return np.mean(correct_confidences)
  
  def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)   # dvalues , dvalues: The derivative of the loss with respect to the output.
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]  #np.eye() is a NumPy function that returns a 2D array representing an identity matrix.
        
        # y_true = np.eye(labels)[y_true] = 
        # [[1, 0, 0],  # index 0
        # [0, 1, 0],  # index 1
        # [0, 0, 1],  # index 2
        # [1, 0, 0]]  # index 0

        # identity_matrix = np.eye(3)
        # [[1. 0. 0.]
        # [0. 1. 0.]
        # [0. 0. 1.]]
        
        # -y_true = [[-1, 0, 0],
        #    [0, -1, 0],
        #    [0, 0, -1],
        #    [-1, 0, 0]]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Calculates the gradient of the loss with respect to the inputs.
        # -y_true represents the error between predicted and true labels.
        # dvalues represents the derivative of the loss with respect to the output.
        # Normalize gradient
        self.dinputs = self.dinputs / samples

        # self.dinputs = 
        # [[(-1/0.1), (0/0.2), (0/0.3)],
        # [(0/0.4), (-1/0.5), (0/0.6)],
        # [(0/0.7), (0/0.8), (-1/0.9)],
        # [(-1/1.0), (0/1.1), (0/1.2)]]

    