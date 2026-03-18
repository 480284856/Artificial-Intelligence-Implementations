import numpy as np

"""
This script implements the Rectified Linear Unit (ReLU) activation function. 
Its primary purpose is to introduce non-linearity into the network by filtering out 
negative values, which allows the model to learn complex patterns.
"""

class ActivationReLu:
    def __init__(self):
        # Initialize variables to store data for backpropagation
        self.inputs = None
        self.output = None

    def forward(self, inputs, training=False):
        """
        Forward pass: Applies the ReLU activation function.
        Logic: If a value is negative, it becomes 0. If it's positive, it stays the same.
        """
        # 1. Save the input values (z-values) for the backward pass later
        self.inputs = inputs

        # 2. ReLU Formula: a = max(0, z). Effectively "extinguishes" negative values.
        self.output = np.maximum(0, inputs)

        return self.output

    def backward(self, delta_incoming): 
        """
        Backward pass: Calculates the gradient (error report) for this layer.
        delta_incoming: The error gradient coming from the next layer (moving right to left).
        """
        # 1. Create a copy of the incoming error report to modify it
        self.delta_outgoing = delta_incoming.copy()

        # 2. Derivative of ReLU: 
        # If the original input was 0 or less, this neuron was "off" and didn't contribute 
        # to the result. Therefore, we set its gradient to 0 to stop the error signal.
        self.delta_outgoing[self.inputs <= 0] = 0 

        return self.delta_outgoing